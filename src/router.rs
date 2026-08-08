use crate::handlers::audio_handler::{reanudar_job, recibir_y_procesar_audio};
use crate::handlers::{jobs_handler, rag_handler};
use crate::mcp;
use crate::state::SharedState;
use axum::{
    Router,
    extract::{DefaultBodyLimit, Request, State},
    http::{
        HeaderValue, Method, StatusCode,
        header::{AUTHORIZATION, CONTENT_TYPE},
    },
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::CorsLayer;

/// Timeout corto para los chequeos de Qdrant/Ollama dentro de `healthcheck` — `/health` no tiene
/// autenticación y el cliente lo llama antes incluso de saber si el backend está vivo, así que no
/// puede quedar colgado esperando a un servicio externo en mal estado. Un puerto simplemente
/// cerrado (`connection refused`) falla casi al instante; este timeout cubre el caso más raro de
/// un puerto filtrado/colgado en vez de rechazado. 3s es generoso frente a lo que tarda un
/// `health_check`/`GET /api/tags` normal en LAN (sub-100ms).
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(3);

/// Un servicio por campo (no `HashMap<String, String>`): son 3 dependencias fijas y conocidas en
/// tiempo de compilación, y `serde(rename)` deja el nombre exacto de cada clave del JSON explícito
/// acá en vez de construido a mano en cada call site. Separar "Ollama Embeddings" de "Ollama
/// Model" en vez de un único "Ollama Server" cuesta lo mismo (una sola llamada a
/// `list_local_models`, ver `chequear_modelos_ollama`) y detecta un caso real que un estado
/// genérico no distingue: el servidor Ollama está arriba pero al modelo configurado
/// (`RAG_EMBEDDING_MODEL`/`RAG_GENERATION_MODEL`) todavía no se le hizo `ollama pull`.
#[derive(Debug, serde::Serialize)]
struct ServiceStatus {
    #[serde(rename = "Qdrant")]
    qdrant: &'static str,
    #[serde(rename = "Ollama Embeddings")]
    ollama_embeddings: &'static str,
    #[serde(rename = "Ollama Model")]
    ollama_model: &'static str,
}

/// `GET /health` — sin autenticación, a propósito: el cliente web lo necesita para distinguir
/// "backend caído/no alcanzable" de "token inválido" antes incluso de mostrar el login. Expone
/// tres cosas: si hay una carga pesada (Whisper o una consulta RAG) corriendo ahora mismo
/// (`heavy_compute_busy`, para que el cliente avise "el servidor está ocupado" antes de mandar una
/// consulta RAG — ver `docs/know_flaws.md`, punto 3), y la disponibilidad real de Qdrant y de los
/// dos modelos de Ollama que el pipeline necesita (`services`) — útil porque el proceso Rust puede
/// estar arriba mientras Qdrant/Ollama no lo están, o mientras el modelo configurado no está
/// instalado, y hoy eso solo se descubre a mitad de un upload/consulta RAG.
#[derive(Debug, serde::Serialize)]
struct HealthResponse {
    status: &'static str,
    heavy_compute_busy: bool,
    services: ServiceStatus,
}

async fn healthcheck(State(estado): State<SharedState>) -> impl IntoResponse {
    let (qdrant_ok, modelos_ollama) =
        tokio::join!(chequear_qdrant(&estado), chequear_modelos_ollama(&estado));

    let (embeddings_ok, model_ok) = match &modelos_ollama {
        Some(modelos) => (
            modelo_instalado(modelos, &estado.config.rag.embedding_model),
            modelo_instalado(modelos, &estado.config.rag.generation_model),
        ),
        None => (false, false),
    };

    axum::Json(HealthResponse {
        status: "ok",
        heavy_compute_busy: estado.heavy_compute_semaphore.available_permits() == 0,
        services: ServiceStatus {
            qdrant: if qdrant_ok { "ok" } else { "error" },
            ollama_embeddings: if embeddings_ok { "ok" } else { "error" },
            ollama_model: if model_ok { "ok" } else { "error" },
        },
    })
}

async fn chequear_qdrant(estado: &SharedState) -> bool {
    matches!(
        tokio::time::timeout(HEALTH_CHECK_TIMEOUT, estado.qdrant.health_check()).await,
        Ok(Ok(_))
    )
}

/// `None` si Ollama no responde o el timeout se cumple (servidor caído/inalcanzable) — en ese caso
/// tanto "Ollama Embeddings" como "Ollama Model" se reportan `error`, ya que sin lista de modelos
/// no hay forma de distinguir "no instalado" de "servidor caído" (y ambos ameritan el mismo aviso
/// al cliente: no se puede confiar en RAG/embeddings ahora mismo).
async fn chequear_modelos_ollama(
    estado: &SharedState,
) -> Option<Vec<ollama_rs::models::LocalModel>> {
    tokio::time::timeout(HEALTH_CHECK_TIMEOUT, estado.ollama.list_local_models())
        .await
        .ok()?
        .ok()
}

/// Compara contra el nombre configurado tolerando el tag `:latest` que Ollama agrega en
/// `list_local_models` aunque `RAG_EMBEDDING_MODEL`/el default no lo incluyan (ej. configurado
/// `bge-m3`, instalado como `bge-m3:latest`) — sin esto, un healthcheck contra un Ollama real y
/// bien configurado reportaría "error" en falso para cualquier modelo sin tag explícito.
fn modelo_instalado(modelos: &[ollama_rs::models::LocalModel], nombre_configurado: &str) -> bool {
    modelos.iter().any(|m| {
        m.name == nombre_configurado
            || m.name
                .strip_suffix(":latest")
                .is_some_and(|base| base == nombre_configurado)
    })
}

/// Límite explícito de tamaño de body para `POST /api/upload-audio` — el default de Axum
/// (`DefaultBodyLimit`, 2 MiB) truncaba en silencio cualquier audio real de este proyecto (audios
/// de varias horas, aunque el códec real sea AMR-NB de bajo bitrate, ~30MB para 5h — ver
/// `CLAUDE.local.md`). 1 GiB deja margen cómodo incluso para el caso más pesado hoy soportado
/// (`.wav` PCM sin comprimir, ~550MB para 5h a 16kHz mono) sin quedar sin límite. Esto NO implica
/// bufferear el archivo en RAM: el handler sigue escribiendo a disco en streaming
/// (`campo.chunk()` + `write_all` por chunk, ver `audio_handler.rs`) — el límite solo acota hasta
/// dónde Axum deja leer el body antes de cortar la request, no cambia cómo se procesa. Scoped
/// solo a esta ruta (no al router completo): las otras rutas con body (`POST /api/search`,
/// `POST /api/rag/answer`, `/mcp`) usan `Json`/JSON-RPC, que sí bufferea el body entero en
/// memoria para parsear — dejarlas en el límite default de 2 MiB evita que un body gigante ahí
/// fuerce una asignación de RAM grande antes de fallar el parseo.
pub(crate) const UPLOAD_BODY_LIMIT_BYTES: usize = 1024 * 1024 * 1024;

/// Token opcional para autenticar el endpoint `/mcp` (ver CLAUDE.local.md: "MCP de solo
/// lectura" — "Bearer token en el endpoint MCP antes de bindear a la IP de LAN"). Leído una sola
/// vez al armar el router, no en cada request.
#[derive(Clone)]
struct McpAuthState {
    token: Option<Arc<str>>,
}

async fn exigir_bearer_token(
    State(auth): State<McpAuthState>,
    req: Request,
    next: Next,
) -> Response {
    let Some(esperado) = &auth.token else {
        // Sin MCP_BEARER_TOKEN configurado: sin autenticación. Aceptable solo mientras el
        // servidor no se expone más allá de localhost — ver advertencia impresa en main.rs.
        return next.run(req).await;
    };

    let autorizado = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|token| token == esperado.as_ref());

    if autorizado {
        next.run(req).await
    } else {
        StatusCode::UNAUTHORIZED.into_response()
    }
}

/// Router de todo lo que requiere el bearer token de `/mcp` (`MCP_BEARER_TOKEN`): el propio
/// `/mcp` y los endpoints REST de lectura (`GET /api/jobs`, `GET /api/jobs/{job_id}`,
/// `GET /api/jobs/{job_id}/transcript`, `GET /api/jobs/{job_id}/metrics`,
/// `GET /api/jobs/{job_id}/audio-segment`, `POST /api/search`, `POST /api/rag/answer`) — todos
/// exponen el mismo contenido transcrito sensible, así que comparten exactamente el mismo
/// middleware/token en vez de un segundo mecanismo de auth. `/api/upload-audio` y
/// `/api/jobs/{id}/resume` (endpoints de escritura ya existentes) quedan fuera de este grupo, sin
/// cambios.
///
/// `.with_state(estado)` pin ea `SharedState` como el estado del router (necesario para los
/// handlers de `rag_handler`, que extraen `State<SharedState>`; los de `jobs_handler` no lo usan
/// pero conviven sin problema en el mismo router) *antes* de aplicar `route_layer` — el middleware
/// de auth (`from_fn_with_state`) recibe `auth_state` de forma explícita y cerrada sobre el
/// closure, independiente del estado propio del router, así que aplicarlo después de fijar
/// `SharedState` no lo afecta.
fn crear_router_protegido(estado: SharedState) -> Router {
    let token = crate::config::get().mcp_bearer_token.clone().map(Arc::from);
    let auth_state = McpAuthState { token };

    Router::new()
        .nest_service("/mcp", mcp::build_service(estado.clone()))
        .route("/api/jobs", get(jobs_handler::listar_jobs_handler))
        .route("/api/jobs/{job_id}", get(jobs_handler::obtener_job_handler))
        .route(
            "/api/jobs/{job_id}/transcript",
            get(jobs_handler::obtener_transcript_handler),
        )
        .route(
            "/api/jobs/{job_id}/metrics",
            get(jobs_handler::obtener_metricas_handler),
        )
        .route(
            "/api/jobs/{job_id}/audio-segment",
            get(jobs_handler::obtener_segmento_audio_handler),
        )
        .route("/api/search", post(rag_handler::buscar_handler))
        .route("/api/rag/answer", post(rag_handler::rag_answer_handler))
        .with_state(estado)
        .route_layer(middleware::from_fn_with_state(
            auth_state,
            exigir_bearer_token,
        ))
}

/// Origen único permitido para CORS (`CLIENT_ORIGIN`, ver `config.rs`) — comportamiento default,
/// sin `cargo run -- --cors-allow-all`. Un valor mal formado en `.env` cae al default
/// (`http://localhost:4321`) en vez de hacer panic al arrancar el servidor; se loguea para que no
/// pase desapercibido.
fn cors_layer_origen_unico() -> CorsLayer {
    let origen = &crate::config::get().services.client_origin;
    let origen = origen.parse::<HeaderValue>().unwrap_or_else(|e| {
        tracing::warn!(
            "CLIENT_ORIGIN '{origen}' no es un origen HTTP válido ({e}), usando el default"
        );
        HeaderValue::from_static("http://localhost:4321")
    });

    CorsLayer::new()
        .allow_origin(origen)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([AUTHORIZATION, CONTENT_TYPE])
}

/// `cargo run -- --cors-allow-all` (ver `docs/api_reference.md`) — acepta peticiones de
/// cualquier origen (`Access-Control-Allow-Origin: *`), para consultar la API desde cualquier
/// dispositivo/origen de la LAN sin fijar `CLIENT_ORIGIN` a un valor único. No reemplaza
/// `MCP_BEARER_TOKEN`: los endpoints protegidos (`/mcp`, `GET /api/jobs*`, `POST /api/search`,
/// `POST /api/rag/answer`) siguen exigiéndolo si está configurado — este flag solo relaja el
/// chequeo de origen que hace el *browser*, no ningún control de acceso propio del servidor.
fn cors_layer_permisivo() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([AUTHORIZATION, CONTENT_TYPE])
}

/// Lee el flag de proceso una sola vez al armar el router (mismo patrón que `--log` en
/// `main.rs`: CLI arg leído directamente vía `std::env::args()`, sin pasar por
/// `Config`/`.env` — ver doc de `WhisperRunner::use_beam_search`).
fn cors_layer() -> CorsLayer {
    if std::env::args().any(|arg| arg == "--cors-allow-all") {
        tracing::warn!(
            "cargo run -- --cors-allow-all activo: el servidor acepta peticiones de CUALQUIER \
             origen (Access-Control-Allow-Origin: *). Esto no reemplaza MCP_BEARER_TOKEN — sin \
             ese token configurado además, cualquier sitio web podría leer las transcripciones."
        );
        cors_layer_permisivo()
    } else {
        cors_layer_origen_unico()
    }
}

pub fn crear_router(estado: SharedState) -> Router {
    let upload = Router::new()
        .route("/api/upload-audio", post(recibir_y_procesar_audio))
        .layer(DefaultBodyLimit::max(UPLOAD_BODY_LIMIT_BYTES))
        .with_state(estado.clone());

    let salud = Router::new()
        .route("/health", get(healthcheck))
        .with_state(estado.clone());

    Router::new()
        .route("/api/jobs/{job_id}/resume", post(reanudar_job))
        .with_state(estado.clone())
        .merge(upload)
        .merge(salud)
        .merge(crear_router_protegido(estado))
        .layer(cors_layer())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tokio::sync::Semaphore;
    use tower::ServiceExt;

    fn estado_de_prueba() -> SharedState {
        Arc::new(AppState {
            nombre_app: "test".to_string(),
            heavy_compute_semaphore: Arc::new(Semaphore::new(1)),
            ollama: ollama_rs::Ollama::default(),
            qdrant: qdrant_client::Qdrant::from_url("http://localhost:6334")
                .build()
                .expect("cliente de Qdrant de prueba"),
            mcp_cancellation_token: tokio_util::sync::CancellationToken::new(),
            config: crate::config::get(),
        })
    }

    #[tokio::test]
    async fn ruta_desconocida_devuelve_404() {
        let app = crear_router(estado_de_prueba());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/no-existe")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// `GET /health` no requiere `Authorization` (a diferencia de `GET /api/jobs` y el resto de
    /// los endpoints protegidos) — el cliente lo necesita para distinguir "backend caído" de
    /// "token inválido" antes de mostrar el login.
    #[tokio::test]
    async fn get_health_devuelve_ok_sin_autenticacion() {
        let app = crear_router(estado_de_prueba());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body_bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["status"], "ok");
        // El semáforo de prueba tiene 1 permiso y nada lo tomó todavía en este test.
        assert_eq!(json["heavy_compute_busy"], false);

        // No se asume si Qdrant/Ollama están corriendo en la máquina donde corre `cargo test` —
        // solo que el chequeo se resuelve (no queda colgado, ver `HEALTH_CHECK_TIMEOUT`) y produce
        // uno de los dos valores posibles para cada servicio.
        for clave in ["Qdrant", "Ollama Embeddings", "Ollama Model"] {
            let valor = json["services"][clave]
                .as_str()
                .unwrap_or_else(|| panic!("services.{clave} no es un string: {json}"));
            assert!(
                valor == "ok" || valor == "error",
                "services.{clave} = '{valor}' inesperado"
            );
        }
    }

    /// Confirma que el `CorsLayer` responde el preflight `OPTIONS` con los headers que el
    /// cliente web necesita para poder hacer `fetch()` desde otro origen — sin esto, el browser
    /// bloquea la request antes de que llegue al handler real, aunque el servidor la aceptaría.
    #[tokio::test]
    async fn preflight_cors_responde_con_headers_de_acceso() {
        let app = crear_router(estado_de_prueba());
        let origen = crate::config::get().services.client_origin.clone();

        let response = app
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/api/jobs")
                    .header("origin", &origen)
                    .header("access-control-request-method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some(origen.as_str())
        );
    }

    /// `cors_layer_permisivo()` (usada cuando `cargo run -- --cors-allow-all` está activo) no se
    /// puede probar a través de `cors_layer()`/`crear_router()` sin reiniciar el proceso de test
    /// con ese argv — se monta directo en un router mínimo en su lugar, confirmando que responde
    /// `Access-Control-Allow-Origin: *` sin importar qué origen mande el cliente.
    #[tokio::test]
    async fn cors_layer_permisivo_acepta_cualquier_origen() {
        let app = Router::new()
            .route("/probe", get(|| async { StatusCode::OK }))
            .layer(cors_layer_permisivo());

        let response = app
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/probe")
                    .header("origin", "http://cualquier-origen-random.example")
                    .header("access-control-request-method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("*")
        );
    }

    #[tokio::test]
    async fn upload_audio_con_content_type_invalido_es_rechazado() {
        let app = crear_router(estado_de_prueba());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/upload-audio")
                    .header("content-type", "text/plain")
                    .body(Body::from("no soy multipart"))
                    .unwrap(),
            )
            .await
            .unwrap();

        // El extractor Multipart de axum rechaza la petición antes de que el handler corra
        // si el content-type no es multipart/form-data con boundary.
        assert!(response.status().is_client_error());
    }

    #[tokio::test]
    async fn resume_de_job_inexistente_devuelve_404() {
        let app = crear_router(estado_de_prueba());
        let job_id = uuid::Uuid::new_v4();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/jobs/{job_id}/resume"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// `job_id` llega crudo desde el path de la URL — un valor que no es un UUID (ej. un intento
    /// de path traversal como `../../etc`) tiene que rechazarse igual que uno bien formado pero
    /// inexistente, nunca usarse tal cual para construir una ruta de disco (ver
    /// `audio_pipeline::job::load_job`).
    #[tokio::test]
    async fn resume_con_job_id_no_uuid_devuelve_404_sin_tocar_disco_fuera_de_jobs() {
        let app = crear_router(estado_de_prueba());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/jobs/no-es-un-uuid/resume")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_job_inexistente_devuelve_404() {
        let app = crear_router(estado_de_prueba());
        let job_id = uuid::Uuid::new_v4();

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/jobs/{job_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_job_con_job_id_no_uuid_devuelve_404() {
        let app = crear_router(estado_de_prueba());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/jobs/no-es-un-uuid")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_transcript_de_job_inexistente_devuelve_404() {
        let app = crear_router(estado_de_prueba());
        let job_id = uuid::Uuid::new_v4();

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/jobs/{job_id}/transcript"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// Un job recién creado (`create_job`) todavía no tiene `transcript.jsonl` en disco — el
    /// endpoint debe responder `200` con `entries: []` y `transcript_ready: false` en vez de
    /// `404`/`500`, según el contrato decidido en docs/TODO.md (el status HTTP no comunica el
    /// estado del dominio, el body sí).
    #[tokio::test]
    async fn get_transcript_de_job_recien_creado_devuelve_entries_vacio() {
        // Mismo lock que `upload_de_archivo_invalido_es_rechazado_sin_crear_job` (ver
        // `audio_pipeline::job::JOBS_DIR_TEST_LOCK`) — este test crea una entrada real en
        // `./jobs`, que sin serializar podría solaparse con el conteo antes/después de ese otro
        // test.
        let _guard = crate::audio_pipeline::job::JOBS_DIR_TEST_LOCK.lock().await;

        let metadata =
            crate::audio_pipeline::job::create_job("wav", None).expect("no se pudo crear el job");
        let app = crear_router(estado_de_prueba());

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/jobs/{}/transcript", metadata.job_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body_bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["transcript_ready"], false);
        assert_eq!(json["entries"].as_array().unwrap().len(), 0);

        let _ = std::fs::remove_dir_all(format!("./jobs/{}", metadata.job_id));
    }

    #[tokio::test]
    async fn get_metrics_de_job_inexistente_devuelve_404() {
        let app = crear_router(estado_de_prueba());
        let job_id = uuid::Uuid::new_v4();

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/jobs/{job_id}/metrics"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// Mismo contrato que `get_transcript_de_job_recien_creado_devuelve_entries_vacio`: un job sin
    /// `metrics.jsonl` en disco todavía responde `200` con `entries: []`, no `404`/`500`.
    #[tokio::test]
    async fn get_metrics_de_job_recien_creado_devuelve_entries_vacio() {
        let _guard = crate::audio_pipeline::job::JOBS_DIR_TEST_LOCK.lock().await;

        let metadata =
            crate::audio_pipeline::job::create_job("wav", None).expect("no se pudo crear el job");
        let app = crear_router(estado_de_prueba());

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/jobs/{}/metrics", metadata.job_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body_bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["entries"].as_array().unwrap().len(), 0);

        let _ = std::fs::remove_dir_all(format!("./jobs/{}", metadata.job_id));
    }

    #[tokio::test]
    async fn get_audio_segment_de_job_inexistente_devuelve_404() {
        let app = crear_router(estado_de_prueba());
        let job_id = uuid::Uuid::new_v4();

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/jobs/{job_id}/audio-segment?start=0&end=10"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// La validación del rango ocurre antes de invocar `ffmpeg` — este test no depende de tener
    /// un audio real en disco ni el binario de `ffmpeg` instalado.
    #[tokio::test]
    async fn get_audio_segment_con_rango_invalido_devuelve_400() {
        let _guard = crate::audio_pipeline::job::JOBS_DIR_TEST_LOCK.lock().await;

        let metadata =
            crate::audio_pipeline::job::create_job("wav", None).expect("no se pudo crear el job");
        let app = crear_router(estado_de_prueba());

        // end <= start
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/api/jobs/{}/audio-segment?start=10&end=5",
                        metadata.job_id
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        // duración > MAX_SEGMENT_SECONDS
        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/api/jobs/{}/audio-segment?start=0&end=999",
                        metadata.job_id
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let _ = std::fs::remove_dir_all(format!("./jobs/{}", metadata.job_id));
    }

    /// No se asevera que el array venga vacío: los tests de este módulo corren en paralelo en el
    /// mismo proceso y comparten el `./jobs` real en disco, así que otro test podría dejar un job
    /// creado en el momento en que este corre. Lo que sí es estable de verificar es que la ruta
    /// responde `200` con un array JSON válido.
    #[tokio::test]
    async fn get_jobs_devuelve_array_json_valido() {
        let jobs_dir = std::path::Path::new("./jobs");
        let _ = std::fs::create_dir_all(jobs_dir);

        let app = crear_router(estado_de_prueba());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/jobs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(json.is_array());
    }

    fn construir_cuerpo_multipart(
        boundary: &str,
        filename: &str,
        content_type: &str,
        contenido: &[u8],
    ) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
        body.extend_from_slice(contenido);
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    fn contar_entradas(dir: &std::path::Path) -> usize {
        std::fs::read_dir(dir).map(|rd| rd.count()).unwrap_or(0)
    }

    /// No depende de audio real ni del modelo — los magic bytes son basura, así que la
    /// validación de Fase 1 debe rechazar antes de siquiera crear un directorio de job.
    ///
    /// Cuenta entradas de `./jobs` antes/después, así que necesita el mismo
    /// `JOBS_DIR_TEST_LOCK` que los tests que crean jobs reales (ver arriba) — sin esto, otro
    /// test corriendo en paralelo en el mismo proceso podría crear una entrada justo en la
    /// ventana entre el conteo "antes" y el "después" y hacer fallar esta aserción sin que haya
    /// ningún bug real.
    #[tokio::test]
    async fn upload_de_archivo_invalido_es_rechazado_sin_crear_job() {
        let _guard = crate::audio_pipeline::job::JOBS_DIR_TEST_LOCK.lock().await;

        let jobs_dir = std::path::Path::new("./jobs");
        let _ = std::fs::create_dir_all(jobs_dir);
        let jobs_antes = contar_entradas(jobs_dir);

        let app = crear_router(estado_de_prueba());
        let boundary = "TESTBOUNDARYINVALIDO";
        let body = construir_cuerpo_multipart(
            boundary,
            "campo_basura.bin",
            "application/octet-stream",
            b"esto no es audio, son bytes cualquiera",
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/upload-audio")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(contar_entradas(jobs_dir), jobs_antes);
    }

    /// El campo `title` se manda ANTES del campo de archivo a propósito — el handler no puede
    /// asumir que el archivo siempre llega primero en el multipart (ver
    /// `handlers::audio_handler::recibir_y_procesar_audio`). También ejercita el trim de
    /// espacios en blanco.
    #[tokio::test]
    async fn upload_con_title_antes_del_archivo_lo_persiste_en_job_json() {
        let _guard = crate::audio_pipeline::job::JOBS_DIR_TEST_LOCK.lock().await;

        let jobs_dir = std::path::Path::new("./jobs");
        let _ = std::fs::create_dir_all(jobs_dir);
        let jobs_antes = contar_entradas(jobs_dir);

        let app = crear_router(estado_de_prueba());
        let boundary = "TESTBOUNDARYTITLE";

        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"title\"\r\n\r\n");
        body.extend_from_slice("Reunión de planificación  ".as_bytes());
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
        body.extend_from_slice(b"RIFF\x24\x08\x00\x00WAVEfmt ");
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/upload-audio")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert_eq!(contar_entradas(jobs_dir), jobs_antes + 1);

        let body_bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        let job_id = json["job_id"].as_str().unwrap().to_string();

        let metadata = crate::audio_pipeline::job::load_job(&job_id).expect("job.json ilegible");
        assert_eq!(metadata.title.as_deref(), Some("Reunión de planificación"));

        let _ = std::fs::remove_dir_all(format!("./jobs/{job_id}"));
    }

    /// Bug real encontrado en la sesión donde se agregó `callback_url`: con el diseño original
    /// (un único campo de archivo leído por fuera de cualquier loop, seguido de un segundo `while`
    /// para los campos restantes), sin un `drop(campo)` explícito después de agotar el campo del
    /// archivo, `Multipart::next_field()` fallaba en silencio con "Error parsing
    /// `multipart/form-data` request" al pedir el siguiente campo — `callback_url` nunca se leía
    /// ni se persistía (verificado manualmente contra el servidor real antes del fix). Tras el
    /// merge con el soporte de `title`, el handler pasó a un único `loop` genérico donde cada
    /// `campo` vive scoped a su iteración — el mismo problema queda evitado por construcción, sin
    /// necesitar ningún `drop` explícito (ver comentario en `recibir_y_procesar_audio`). Este test
    /// sigue siendo la regresión que cubre el escenario real: un multipart con el campo de
    /// archivo seguido de `callback_url`, tal como lo arma `UploadForm` en el cliente web.
    #[tokio::test]
    async fn upload_con_callback_url_lo_persiste_en_job_json() {
        let _guard = crate::audio_pipeline::job::JOBS_DIR_TEST_LOCK.lock().await;

        let app = crear_router(estado_de_prueba());
        let boundary = "TESTBOUNDARYCALLBACK";
        // Header WAV válido (pasa el sniff de magic bytes) pero sin datos reales — no hace falta
        // que decodifique bien, esto solo prueba el parseo de los campos del multipart, no el
        // pipeline de Whisper.
        let wav_minimo: &[u8] = b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\
             \x80\xBB\x00\x00\x00\xEE\x02\x00\x02\x00\x10\x00data\x00\x08\x00\x00";

        let mut body =
            construir_cuerpo_multipart(boundary, "callback.wav", "audio/wav", wav_minimo);
        // `construir_cuerpo_multipart` ya cierra el body con el boundary final — hay que sacarle
        // ese cierre para poder agregar el campo `callback_url` y recién ahí volver a cerrarlo.
        let cierre = format!("--{boundary}--\r\n");
        assert!(body.ends_with(cierre.as_bytes()));
        body.truncate(body.len() - cierre.len());
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"callback_url\"\r\n\r\n");
        body.extend_from_slice(b"http://127.0.0.1:9/webhook-test");
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(cierre.as_bytes());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/upload-audio")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::ACCEPTED);

        let body_bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        let job_id = json["job_id"].as_str().unwrap().to_string();

        // `callback_url` se persiste sincrónicamente en el handler, antes de disparar el
        // procesamiento en background — ya está en disco para cuando llega la respuesta 202, sin
        // necesidad de esperar ni pollear.
        let job_json_path = format!("./jobs/{job_id}/job.json");
        let contenido = std::fs::read_to_string(&job_json_path).expect("job.json debería existir");
        let job_json: serde_json::Value = serde_json::from_str(&contenido).unwrap();
        assert_eq!(job_json["callback_url"], "http://127.0.0.1:9/webhook-test");

        let _ = std::fs::remove_dir_all(format!("./jobs/{job_id}"));
    }

    /// Ruta del audio real usado por el test de punta a punta — configurable vía la variable de
    /// entorno `TEST_AUDIO_PATH` (no vía argumento de `cargo test`, que el harness de test
    /// integrado de Rust no expone al binario) para poder probar con cualquier archivo de
    /// `sample_Media/` sin tocar código. Por defecto usa la muestra corta (~2 min) para que la
    /// corrida sea rápida durante el desarrollo de las fases siguientes.
    fn ruta_audio_de_prueba() -> String {
        std::env::var("TEST_AUDIO_PATH")
            .unwrap_or_else(|_| "sample_Media/Muestra2_02min.m4a".to_string())
    }

    /// Content-type + nombre de archivo acordes a la extensión real, solo por prolijidad del
    /// multipart de prueba — el handler nunca confía en esto, sniffea magic bytes (ver
    /// `audio_handler.rs`).
    fn content_type_y_nombre(path: &str) -> (&'static str, String) {
        let filename = std::path::Path::new(path)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("audio")
            .to_string();
        let content_type = if path.to_lowercase().ends_with(".mp3") {
            "audio/mpeg"
        } else {
            "audio/mp4"
        };
        (content_type, filename)
    }

    /// Test de punta a punta con un archivo real: sube el audio de `ruta_audio_de_prueba()` a
    /// través del endpoint real, espera a que el pipeline completo (decode + resample + chunk +
    /// whisper + persistencia JSONL, Fase 2/3) escriba una transcripción, y además confirma que
    /// la Fase 4 (embeddings) encadenada después del pipeline (ver `audio_handler.rs`) terminó
    /// insertando puntos en Qdrant para este job — es el primer test que ejercita las fases entre
    /// sí en vez de aisladas, base para construir el retrieval de Fase 5 encima.
    ///
    /// Ignorado por defecto porque depende de piezas que no están versionadas ni siempre
    /// levantadas: `sample_Media/` (audio real del usuario), `models/ggml-small-q5_1.bin` (el
    /// modelo GGML), y Ollama (`bge-m3`) + Qdrant reales corriendo en localhost. Correr con
    /// `cargo test pipeline_hardcodeado -- --ignored --nocapture`, o con otra muestra:
    /// `TEST_AUDIO_PATH="sample_Media/otro.m4a" cargo test pipeline_hardcodeado -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "requiere sample_Media/ real, models/ggml-small-q5_1.bin, y Ollama+Qdrant reales"]
    async fn pipeline_hardcodeado_transcribe_audio_real() {
        let audio_path = ruta_audio_de_prueba();

        let audio_bytes = std::fs::read(&audio_path)
            .unwrap_or_else(|e| panic!("no se pudo leer {audio_path}: {e}"));

        let app = crear_router(estado_de_prueba());
        let boundary = "TESTBOUNDARYREAL";
        let (content_type, filename) = content_type_y_nombre(&audio_path);
        let body = construir_cuerpo_multipart(boundary, &filename, content_type, &audio_bytes);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/upload-audio")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::ACCEPTED);

        let body_bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        let job_id = json["job_id"]
            .as_str()
            .expect("respuesta sin job_id")
            .to_string();

        let transcript_path = format!("./jobs/{job_id}/transcript.jsonl");

        let mut intentos = 0;
        loop {
            if let Ok(contenido) = std::fs::read_to_string(&transcript_path)
                && !contenido.trim().is_empty()
            {
                println!("Transcript generado ({transcript_path}):\n{contenido}");
                assert!(contenido.lines().count() >= 1);
                break;
            }
            intentos += 1;
            assert!(
                intentos < 150,
                "timeout esperando la transcripción en {transcript_path}"
            );
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }

        // Fase 4 corre encadenada justo después de Fase 2/3 (ver audio_handler.rs) — confirmar
        // que los embeddings de este job realmente llegaron a Qdrant es lo que hace de este test
        // una prueba de los módulos entre sí, no solo de Whisper en aislamiento.
        let qdrant = qdrant_client::Qdrant::from_url("http://localhost:6334")
            .build()
            .expect("cliente de Qdrant de prueba");
        let filter =
            qdrant_client::qdrant::Filter::must([qdrant_client::qdrant::Condition::matches(
                "audio_id",
                job_id.clone(),
            )]);

        let mut intentos = 0;
        loop {
            let count = qdrant
                .count(
                    qdrant_client::qdrant::CountPointsBuilder::new("transcripts")
                        .filter(filter.clone())
                        .exact(true),
                )
                .await
                .expect("no se pudo contar los puntos del job en Qdrant")
                .result
                .expect("respuesta de count sin resultado")
                .count;

            if count > 0 {
                println!("Embeddings del job {job_id} encontrados en Qdrant: {count} puntos");
                break;
            }

            intentos += 1;
            assert!(
                intentos < 60,
                "timeout esperando los embeddings del job {job_id} en Qdrant"
            );
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }

        // Limpieza: no dejar puntos de este job de prueba en la colección real.
        let _ = qdrant
            .delete_points(
                qdrant_client::qdrant::DeletePointsBuilder::new("transcripts").points(filter),
            )
            .await;
    }
}
