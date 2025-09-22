use ndarray::Array2;
use ort::{Session, Error};
use tokenizers::Tokenizer;

pub fn classify_text(classifier: &Session, tokenizer: &Tokenizer, inputs: Vec<&str>) -> Vec<Vec<f32>> {

	// Load the tokenizer and encode the text.
	let input_len = inputs.len();
	// Encode our input strings. `encode_batch` will pad each input to be the same length.
	let encodings = tokenizer.encode_batch(inputs, false).map_err(|e| Error::new(e.to_string())).unwrap();

	// Get the padded length of each encoding.
	let padded_token_length = encodings[0].len();

	// Get our token IDs & mask as a flattened array.
	let ids: Vec<i64> = encodings.iter().flat_map(|e| e.get_ids().iter().map(|i| *i as i64)).collect();
	let mask: Vec<i64> = encodings.iter().flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64)).collect();
	let types: Vec<i64> = encodings.iter().flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64)).collect();

	// Convert our flattened arrays into 2-dimensional tensors of shape [N, L].
	let a_ids = Array2::from_shape_vec([input_len, padded_token_length], ids).unwrap();
	let a_mask = Array2::from_shape_vec([input_len, padded_token_length], mask).unwrap();
	let a_types = Array2::from_shape_vec([input_len, padded_token_length], types).unwrap();

	// Run the model.
	let outputs = classifier.run(ort::inputs![a_ids, a_mask, a_types].unwrap()).unwrap()["logits"].try_extract_tensor::<f32>().unwrap().into_owned();

	// Extract our embeddings tensor and convert it to a strongly-typed 2-dimensional array.
	outputs.into_raw_vec_and_offset().0.chunks(2).collect::<Vec<&[f32]>>().iter().map(|&e| e.to_vec()).collect::<Vec<Vec<f32>>>()
}

pub fn classify_text_warmup (classifier: &Session) {
	let input_tensor = ort::Tensor::from_array(([1usize, 20], vec![0i64; 20])).unwrap();
	let input_tensor2 = ort::Tensor::from_array(([1usize, 20], vec![0i64; 20])).unwrap();
	let input_tensor3 = ort::Tensor::from_array(([1usize, 20], vec![0i64; 20])).unwrap();
	classifier.run(ort::inputs![input_tensor, input_tensor2, input_tensor3].unwrap()).unwrap()["logits"].try_extract_tensor::<f32>().unwrap().into_owned();
}