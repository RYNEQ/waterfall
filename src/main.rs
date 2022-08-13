use image::{ImageBuffer, Luma};


use std::path::Path;
use std::ffi::OsStr;
use std::env;
use std::io::BufReader;
use std::fs::File;
use byteorder::{LittleEndian, ReadBytesExt};
use dsp::num_complex::Complex32;
use dsp::runtime::node::ProcessNode;
use dsp::node::fft::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        println!("Usage: {} <file.raw> <samplerate> <outfile.png>", args[0]);
        return;
    }
    const FFT_SIZE: usize = 1024;
    let file_name = &args[1];
    let sample_rate = args[2].parse::<f32>().unwrap();
    let outfile = &args[3];


    let file = File::open(file_name).unwrap();
    let mut reader = BufReader::with_capacity(8192, file);
    let mut fft = ForwardFFT::new(FFT_SIZE, WindowType::Hamming);

    let mut spectrum: Vec<Vec<f32>> = vec![];

    let mut done = false;
    let mut sample_count = 0;
    let mut min_value = f32::MAX;
    let mut max_value = f32::MIN;
    while !done {
        let mut buffer = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
        let mut fft_res = vec![Complex32::new(0.0, 0.0); FFT_SIZE];

        for idx in 0..FFT_SIZE {
            let i = reader.read_f32::<LittleEndian>(); 
            let q = reader.read_f32::<LittleEndian>();
            match (i,q){
                (Ok(i), Ok(q)) => {
                    buffer[idx] = Complex32::new(i, q);
                    sample_count += 1;
                },
                _ => {
                    done = true;
                    break;
                }
            }
        }
        fft.process_buffer(&buffer, &mut fft_res).unwrap();
        let mut fft_norm: Vec<f32> = fft_res.into_iter().map(|c| {
            10.0*f32::log10(c.norm().powf(2.0))
        }).collect();
        fft_norm.rotate_right(FFT_SIZE/2);

        min_value = fft_norm.iter().fold(min_value, |a,b| a.min(*b));
        max_value = fft_norm.iter().fold(max_value, |a,b| a.max(*b));
        spectrum.push(fft_norm);
    }
    println!("{}, {}", min_value, max_value);
    let image: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_fn(FFT_SIZE.try_into().unwrap(), spectrum.len().try_into().unwrap(), |x, y| {
        let val = ((spectrum[y as usize][x as usize]  - min_value) / (max_value - min_value))*255.0;
        Luma([val as u8])
    });
    image.save(outfile).unwrap();

    let path = Path::new(outfile);
    let stem = path.file_stem().unwrap_or(OsStr::new(""));
    let mut outfile_resized = stem.to_os_string();
    outfile_resized.push("_resized.");
    if let Some(extension) = path.extension() {
        outfile_resized.push(extension);
    }
    let outfile_resized = path.parent().unwrap().join(outfile_resized);

    image::imageops::resize(&image, FFT_SIZE as u32, (sample_count as f32 /sample_rate as f32) as u32, image::imageops::FilterType::Nearest).save(outfile_resized).unwrap();
    println!("Samples: {:?}", sample_count);
    println!("Spectrum Height: {:?}", spectrum.len());

}
