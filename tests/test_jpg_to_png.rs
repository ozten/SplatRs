//! Minimal JPG->PNG test

#[test]
#[ignore]
fn minimal_jpg_to_png() {
    // Absolute minimal conversion
    let jpg = image::open(concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/dollhouse_sm/input/IMG_0910.jpg"))
        .unwrap()
        .to_rgb8();

    jpg.save("/tmp/minimal_conversion.png").unwrap();
    println!("Saved /tmp/minimal_conversion.png");

    // Also try saving as different format
    jpg.save("/tmp/minimal_conversion.tiff").unwrap();
    println!("Saved /tmp/minimal_conversion.tiff");
}
