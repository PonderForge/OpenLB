use ort::CPUExecutionProvider;
use openlb::text_filter::TxtCleaner;

fn main () {
    let txtcleaner = TxtCleaner::init(Some(0.6), Some(CPUExecutionProvider::default().into()));
    txtcleaner.warmup(20);
    let cleaned_text = txtcleaner.clean_text("I like trains!".to_string());
    println!("Cleaned Text: {}", cleaned_text);
}