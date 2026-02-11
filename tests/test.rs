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
    
    let now = Instant::now();
    let out = cleaner.clean_image(img, ImgCleanLevel::Human);
    println!("{:?}", now.elapsed());
    if out.is_none() { 
        println!("No NSFW Content Detected");
        return;
    }
    let out = out.unwrap();
    out.save("out.jpg").unwrap();
}

#[test]
#[cfg(feature = "human_scan")]
fn clean_humans() {
    let cleaner = ImgCleaner::builder().with_overlay_type(openlb::img_filter::ImgCleanOverlay::Blur).commit();
    let img = image::open("test.jpg").unwrap();
    
    let now = Instant::now();
    let out = cleaner.clean_image(img, ImgCleanLevel::Human);
    println!("{:?}", now.elapsed());
    if out.is_none() { 
        println!("No NSFW Content Detected");
        return;
    }
    let out = out.unwrap();
    out.save("out.jpg").unwrap();
}

#[test]
#[cfg(feature = "text_scan")]
fn clean_text() {
    let txtcleaner = TxtCleaner::builder().commit();
    let now = Instant::now();
    std::fs::write("out.txt", txtcleaner.clean_text(std::fs::read_to_string("test.txt").unwrap())).unwrap();
    println!("Text Detect Time: {:?}", now.elapsed());
}

#[test]
#[cfg(feature = "image_scan")]
fn clean_folder() {
    let cleaner = ImgCleaner::builder().commit();
    let paths = std::fs::read_dir("./test").unwrap();
    let mut i = 0;
    for path in paths {
        let p = path.unwrap().path();
        let path = p.as_os_str().to_str().unwrap();
        if path.ends_with("jpg") {
            let img = image::open(path).unwrap();
            let out = cleaner.clean_image(img.clone(), ImgCleanLevel::Human);
            if out.is_none() {
                img.save(&format!("out/out{}.png", i)).unwrap();
                i+=1; 
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
#[cfg(feature = "human_scan")]
fn humans_time() {

    let cleaner = ImgCleaner::builder().with_overlay_type(openlb::img_filter::ImgCleanOverlay::Icon).commit();
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
    let text = "";
    let now = Instant::now();
    for _ in 0..50 {
        txtcleaner.clean_text(&text);
    }
    println!("Average Time: {:?}", now.elapsed() / 50);
    println!("{}", txtcleaner.clean_text(&text));
}

#[test]
#[cfg(feature = "gif")]
fn clean_gif() {
    use std::io::Write;

    let cleaner = ImgCleaner::builder().with_min_human_size(100).commit();
    let input = std::fs::File::open("gif_test/test.gif").unwrap();
    let out = cleaner.clean_gif_nd(input);
    let mut file = std::fs::File::create("out.gif").unwrap();
    if out.is_none() {
        println!("No NSFW Content Detected");
    } else {
        file.write(&out.unwrap()).unwrap();
    }
}