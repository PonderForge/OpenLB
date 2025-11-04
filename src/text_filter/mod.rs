
mod classifier;
use classifier::{classify_string, classify_string_warmup};

use ort::{ExecutionProviderDispatch, GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;

use fancy_regex::Regex;
pub struct TxtCleanerBuilder {
    sentence_threshold: f32,
    exec_provider: ExecutionProviderDispatch
}

impl TxtCleanerBuilder {
    pub fn with_sentence_thres (mut self, threshold: f32) -> TxtCleanerBuilder {
        self.sentence_threshold = threshold;
        self
    }

    pub fn with_exec_provider (mut self, provider: ExecutionProviderDispatch) -> TxtCleanerBuilder {
        self.exec_provider = provider;
        self
    }

    pub fn commit (self) -> TxtCleaner {
        let ort_init = ort::init().with_execution_providers([self.exec_provider]).commit();
        if ort_init.is_err() {
            panic!("ONNX was not correctly initalized!");
        }

        //Load Models
        let classifier = Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap().commit_from_file("./models/text_classify.onnx").unwrap();
        let tokenizer: Tokenizer = Tokenizer::from_file("./models/text_tokenizer.json").unwrap();
        for _ in 0..20 {
            classify_string_warmup(&classifier);
        }
        let re = Regex::new(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s").unwrap();
        TxtCleaner { classifier: classifier, sentence_threshold: self.sentence_threshold, tokenizer: tokenizer, sentence_seperator: re}
    }
}

#[derive(Debug)]
pub struct TxtCleaner {
    classifier: Session,
    tokenizer: Tokenizer,
    sentence_threshold: f32,
    sentence_seperator: Regex
}

impl TxtCleaner {

    pub fn builder() -> TxtCleanerBuilder {
        TxtCleanerBuilder {sentence_threshold: 0.8, exec_provider: ort::CPUExecutionProvider::default().into()}
    }

    pub fn clean_text<S: AsRef<str>>(&self, text: S) -> String {
        let mut sentences: Vec<Vec<&str>> = self.sentence_seperator.split(text.as_ref()).map(|x| x.unwrap()).collect::<Vec<&str>>().chunks(100).collect::<Vec<&[&str]>>().iter().map(|&e| e.to_vec()).collect::<Vec<Vec<&str>>>();
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
        sentences.into_iter().flatten().collect::<Vec<&str>>().join(" ")
    }

    pub fn classify_text<S: AsRef<str>>(&self, text: S) -> Vec<Vec<f32>> {
        let mut sentences: Vec<Vec<&str>> = self.sentence_seperator.split(text.as_ref()).map(|x| x.unwrap()).collect::<Vec<&str>>().chunks(100).collect::<Vec<&[&str]>>().iter().map(|&e| e.to_vec()).collect::<Vec<Vec<&str>>>();
        let mut returns: Vec<Vec<f32>> = Vec::new();
        for sentence_group in sentences.iter_mut() {
            let mut ret = classify_string(&self.classifier, &self.tokenizer, sentence_group.clone());
            returns.append(&mut ret);
        }
        returns
    }
}
