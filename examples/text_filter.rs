use openlb::text_filter::TxtCleaner;

fn main () {
    let txtcleaner = TxtCleaner::builder().commit();
    let cleaned_text = txtcleaner.clean_text("I like trains!".to_string());
    println!("Cleaned Text: {}", cleaned_text);
}