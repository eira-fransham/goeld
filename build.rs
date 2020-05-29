use std::{env, fs, path::PathBuf};

const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");
const SHADER_EXTENSIONS: &[&str] = &["vert", "frag"];

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    let mut compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    options.set_include_callback(|name, include_type, _parent, _depth| match include_type {
        shaderc::IncludeType::Relative => {
            let out = fs::read_to_string(name).map_err(|e| e.to_string())?;

            Ok(shaderc::ResolvedInclude {
                resolved_name: name.to_string(),
                content: out,
            })
        }
        shaderc::IncludeType::Standard => {
            let path = PathBuf::from(SHADER_DIR).join(name);
            let out = fs::read_to_string(path).map_err(|e| e.to_string())?;

            Ok(shaderc::ResolvedInclude {
                resolved_name: name.to_string(),
                content: out,
            })
        }
    });

    for shader in fs::read_dir(SHADER_DIR).unwrap() {
        let path = shader.unwrap().path();

        if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
            if !SHADER_EXTENSIONS.contains(&ext) {
                continue;
            }
        } else {
            continue;
        }

        let name = path.file_name().unwrap().to_str().unwrap().to_string() + ".spv";
        let dest_path = out_dir.join(name);

        println!("{}", path.display());
        let bin = compiler
            .compile_into_spirv(
                &fs::read_to_string(&path).unwrap(),
                shaderc::ShaderKind::InferFromSource,
                &path.display().to_string(),
                "main",
                Some(&options),
            )
            .unwrap();

        fs::write(dest_path, bin.as_binary_u8()).unwrap();

        println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shaders");
}
