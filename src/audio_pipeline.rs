use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Extrae el texto de un archivo de audio pesado leyendo en fragmentos.
/// Retorna un vector de strings para mantener la huella de memoria al mínimo.
pub fn extraer_texto_aislado(ruta_archivo: String) -> Vec<String> {
    println!("Iniciando pipeline de Whisper para: {}", ruta_archivo);
    
    // 1. Cargar el modelo Cuantizado. 
    // Nota: El archivo .bin debe estar en esta ruta relativa.
    let ruta_modelo = "./modelos/ggml-base-tdrz-q5_1.bin";
    
    // Inicialización robusta para la versión 0.16.0
    let ctx_params = WhisperContextParameters::default();
    let ctx = WhisperContext::new_with_params(ruta_modelo, ctx_params)
        .expect("Error crítico: No se pudo cargar el modelo de Whisper al contexto.");
        
    let mut state = ctx.create_state().expect("Error al crear el estado del modelo.");

    // 2. Configuración estricta de parámetros para evitar alucinaciones con ruido
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_no_context(true); // Ignora el contexto anterior para no repetir frases en zonas de estática
    // params.set_entropy_threshold(2.4);
    params.set_suppress_blank(true);
    
    // Si usas un modelo tdrz (TinyDiarize), descomenta esta línea para detectar hablantes
    // params.set_tdrz_enable(true);

    let mut fragmentos_transcritos = Vec::new();

    // 3. Procesamiento en ventanas (Chunking)
    // En un flujo de producción completo, aquí usaríamos la librería `symphonia` 
    // para decodificar el archivo a 16kHz y entregar arrays de muestras dinámicamente.
    // Aquí implementamos la estructura lógica iterativa:
    
    let numero_de_chunks_simulados = 5; // Representa los fragmentos que symphonia entregaría
    
    for chunk_idx in 0..numero_de_chunks_simulados {
        // Un chunk estándar de 30 segundos a 16kHz contiene 480,000 muestras f32.
        // Esto ocupa apenas ~1.9 MB en RAM, protegiendo totalmente nuestro sistema.
        let chunk_audio: Vec<f32> = vec![0.0; 480_000]; 
        
        // Ejecutar inferencia en este bloque específico
        state.full(params.clone(), &chunk_audio)
            .expect("Fallo en la inferencia de Whisper para el chunk actual.");
        
        let mut texto_chunk = String::new();
        let num_segmentos = state.full_n_segments();
        
        for i in 0..num_segmentos {
            if let Some(segmento ) = state.get_segment(i) {
                if let Ok(texto) = segmento.to_str_lossy() {
                    texto_chunk.push_str(&texto);
                    texto_chunk.push_str(" ");
                }
            }
        }
        
        let texto_limpio = texto_chunk.trim();
        if !texto_limpio.is_empty() {
            fragmentos_transcritos.push(texto_limpio.to_string());
        }
        
        println!("Fragmento {} transcrito exitosamente.", chunk_idx + 1);
    }

    // 4. CRÍTICO: Liberación explícita para salvar la RAM
    // Obligamos a Rust a invocar el destructor de C++ de whisper.cpp inmediatamente.
    drop(state);
    drop(ctx);
    println!("Fase de transcripción completa. Modelo Whisper descargado de la memoria RAM.");

    fragmentos_transcritos
}