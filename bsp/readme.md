# Example usage

(Assuming that the pk3 file is unzipped locally)

```rust
extern crate bsp;

use std::fs::File;

fn main() -> std::io::Result<()> {
    let bsp = bsp::read_bsp(&File::open("pak0/maps/q3dm1.bsp")?)?;
    println!("{:?}", bsp);

    Ok(())
}
```
