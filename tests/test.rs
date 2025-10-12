#[cfg(feature = "text_scan")]
use openlb::text_filter::TxtCleaner;
use std::time::Instant;
#[cfg(feature = "image_scan")]
use openlb::img_filter::{ImgCleanLevel, ImgCleaner};


#[test]
#[cfg(feature = "image_scan")]
fn clean_image() {
    let cleaner = ImgCleaner::builder().commit();
    let img = image::open("test.jpg").unwrap();
    
    let out = cleaner.clean_image(img, ImgCleanLevel::Human);
    if out.is_none() {
        println!("No NSFW Content Detected");
        return;
    }
    let out = out.unwrap();
    out.save("out.jpg");
}

#[test]
#[cfg(feature = "text_scan")]
fn clean_text() {
    let txtcleaner = TxtCleaner::builder().commit();
    let now = Instant::now();
    std::fs::write("out.txt", txtcleaner.clean_text(std::fs::read_to_string("test.txt").unwrap()));
    println!("Text Detect Time: {:?}", now.elapsed());
}

#[test]
#[cfg(feature = "image_scan")]
fn clean_folder() {
    let cleaner = ImgCleaner::builder().commit();
    let paths = std::fs::read_dir("./ai_nsfw_test").unwrap();
    let mut i = 0;
    for path in paths {
        let p = path.unwrap().path();
        let path = p.as_os_str().to_str().unwrap();
        if path.ends_with("jpg") {
            let out = cleaner.clean_image(image::open(path).unwrap(), ImgCleanLevel::Human);
            if out.is_none() {
                println!("No NSFW Content Detected");
            } else {
                let out = out.unwrap();
                out.save(&format!("out/out{}.png", i)).unwrap();
                i+=1; 
            }
        }
    }
    println!("Finished");
}

#[test]
#[cfg(feature = "image_scan")]
fn image_time() {

    let cleaner = ImgCleaner::builder().commit();
    let input_img = image::open("test.jpg").unwrap();
    let mut tot_time = 0;
    for _ in 0..50 {
        let img = input_img.clone();
        let now = Instant::now();
        cleaner.classify_image(img, ImgCleanLevel::Human);
        tot_time += now.elapsed().as_millis();
    }
    println!("Average Time: {:?}ms", tot_time / 50);
}

#[test]
#[cfg(feature = "text_scan")]
fn text_time() {
    let txtcleaner = TxtCleaner::builder().commit();
    let text: String = "Hello I like trains! I love Jesus! How much would a wood chuck chuck if a wood chuck would chuck wood? The girl put on her leotard and let him stroke her.".to_string();
    let now = Instant::now();
    for _ in 0..50 {
        txtcleaner.clean_text(&text);
    }
    println!("Average Time: {:?}", now.elapsed() / 50);
}

#[test]
#[cfg(feature = "gif")]
fn clean_gif() {
    use std::io::Write;

    let thresholds = LBThresholds { sexy: 0.1, porn: 0.74, hentai: 0.5 };
    let cleaner = ImgCleaner::init(Some(thresholds), Some(thresholds), Some(CPUExecutionProvider::default().into()));
    cleaner.warmup(20);
    let input = std::fs::File::open("unnamed.gif").unwrap();
    let out = cleaner.clean_gif(input);
    let mut file = std::fs::File::create("out.gif").unwrap();
    if out.is_none() {
        println!("No NSFW Content Detected");
    } else {
        file.write(&out.unwrap()).unwrap();
    }
}