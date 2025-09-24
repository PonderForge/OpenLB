
mod classifier;
use classifier::{classify_string, classify_string_warmup};

use ort::{ExecutionProviderDispatch, GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;

#[derive(Debug)]
#[cfg(feature = "text_scan")]
pub struct TxtCleaner {
    classifier: Session,
    tokenizer: Tokenizer,
    sentence_threshold: f32
}

#[cfg(feature = "text_scan")]
impl TxtCleaner {
    pub fn init(threshold: Option<f32>, exec_providers: Option<ExecutionProviderDispatch>) -> TxtCleaner {
        //Initialize Onnx Runtime
        let ort_init = ort::init().with_execution_providers([exec_providers.unwrap_or_else(||{ort::CPUExecutionProvider::default().into()})]).commit();
        if ort_init.is_err() {
            panic!("ONNX was not correctly initalized!");
        }

        //Load Models
        let classifier = Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap().commit_from_memory(include_bytes!("../../models/text_classify.onnx")).unwrap();
        let tokenizer: Tokenizer = Tokenizer::from_bytes(include_bytes!("../../models/text_tokenizer.json")).unwrap();
        TxtCleaner { classifier: classifier, sentence_threshold: threshold.unwrap_or_else(||{0.8}), tokenizer: tokenizer}
    }

    pub fn warmup(&self, iters: u8) {
        //Warmup Models
        for _ in 0..iters {
            classify_string_warmup(&self.classifier);
        }
    }

    pub fn clean_text<S: AsRef<str>>(&self, text: S) -> String {
        let mut sentences: Vec<Vec<&str>> = text.as_ref().split_inclusive(['.', '?', '!']).collect::<Vec<&str>>().chunks(100).collect::<Vec<&[&str]>>().iter().map(|&e| e.to_vec()).collect::<Vec<Vec<&str>>>();
        for sentence_group in sentences.iter_mut() {
            let ret = &classify_string(&self.classifier, &self.tokenizer, sentence_group.clone());
            let mut removals = 0;
            for i in 0..sentence_group.len() {
                if ret[i][1] > self.sentence_threshold {
                    sentence_group.remove(i - removals);
                    removals += 1;
                }
            }
        }
        sentences.into_iter().flatten().collect::<Vec<&str>>().join("")
    }

    pub fn classify_text<S: AsRef<str>>(&self, text: S) -> Vec<Vec<f32>> {
        let mut sentences: Vec<Vec<&str>> = text.as_ref().split_inclusive(['.', '?', '!']).collect::<Vec<&str>>().chunks(100).collect::<Vec<&[&str]>>().iter().map(|&e| e.to_vec()).collect::<Vec<Vec<&str>>>();
        let mut returns: Vec<Vec<f32>> = Vec::new();
        for sentence_group in sentences.iter_mut() {
            let mut ret = classify_string(&self.classifier, &self.tokenizer, sentence_group.clone());
            returns.append(&mut ret);
        }
        returns
    }
}
