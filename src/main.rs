use clap::{App, Arg};

use std::{
    fs::File,
    io::{prelude::*, BufRead, BufReader},
    path::PathBuf,
};

use lang::{parse_program, ParseError};

pub(self) mod lang;

fn main() -> Result<(), std::boxed::Box<dyn std::error::Error>> {
    simple_logger::init().unwrap();

    let app = App::new("tiger")
        .version("0.1.0")
        .author("Favil Orbedios")
        .about("Tiger language interpreter")
        .arg(Arg::with_name("input").help("input file").required(true));
    let matches = app.get_matches();
    let input_file = PathBuf::from(matches.value_of("input").unwrap());
    log::debug!("Interpretting {}", input_file.to_str().unwrap());
    let mut f = File::open(input_file)?;
    let mut contents = String::new();
    f.read_to_string(&mut contents)?;
    let p = parse_program(&contents);
    log::info!("{:?}", p);
    Ok(())
}
