use ndarray::Array2;
use ort::{Session, Error};
use tokenizers::Tokenizer;

pub fn classify_string(classifier: &Session, tokenizer: &Tokenizer, inputs: Vec<&str>) -> Vec<Vec<f32>> {

	let input_len = inputs.len();
	let encodings = tokenizer.encode_batch(inputs, false).map_err(|e| Error::new(e.to_string())).unwrap();
	let padded_token_length = encodings[0].len();
	let ids: Vec<i64> = encodings.iter().flat_map(|e| e.get_ids().iter().map(|i| *i as i64)).collect();
	let mask: Vec<i64> = encodings.iter().flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64)).collect();
	let types: Vec<i64> = encodings.iter().flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64)).collect();
	let a_ids = Array2::from_shape_vec([input_len, padded_token_length], ids).unwrap();
	let a_mask = Array2::from_shape_vec([input_len, padded_token_length], mask).unwrap();
	let a_types = Array2::from_shape_vec([input_len, padded_token_length], types).unwrap();
	let outputs = classifier.run(ort::inputs![a_ids, a_mask, a_types].unwrap()).unwrap()["logits"].try_extract_tensor::<f32>().unwrap().into_owned();
	outputs.into_raw_vec_and_offset().0.chunks(2).collect::<Vec<&[f32]>>().iter().map(|&e| e.to_vec()).collect::<Vec<Vec<f32>>>()
}

pub fn classify_string_warmup (classifier: &Session) {
	let blank_tensor = Array2::from_shape_vec([1usize, 128], vec![0i64; 128]).unwrap();
	classifier.run(ort::inputs![blank_tensor.clone(), blank_tensor.clone(), blank_tensor].unwrap()).unwrap()["logits"].try_extract_tensor::<f32>().unwrap().into_owned();
}