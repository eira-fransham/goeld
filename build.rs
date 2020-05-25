use std::{env, ffi::OsStr, fs, path::PathBuf, process};

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut commands = vec![];

    for shader in fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders")).unwrap() {
        let path = shader.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_string() + ".spv";
        let dest_path = out_dir.join(name);

        // So the build fails if `glslangValidator` fails
        let _ = fs::remove_file(&dest_path);

        commands.push(
            process::Command::new("glslangValidator")
                .args(&[
                    OsStr::new("-V"),
                    path.as_ref(),
                    OsStr::new("-o"),
                    dest_path.as_ref(),
                ])
                .spawn()
                .unwrap(),
        );

        println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
    }

    for mut c in commands {
        if !c.wait().unwrap().success() {
            panic!("Shader compilation failed");
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shaders");
}
