use clap::Parser;
use color_eyre::{eyre::bail, Result};
use miette::GraphicalReportHandler;
use nom_tracable::{cumulative_histogram, histogram};

use std::{fs::File, io::prelude::*, path::PathBuf};

use crate::lang::Program;

mod error;
mod lang;

#[derive(Parser, Debug)]
#[command(author="Favil Orbedios", version="0.1.0", about="Tiger language interpreter/compiler", long_about=None)]
struct Args {
    input: PathBuf,
}

fn main() -> Result<()> {
    simple_logger::init().unwrap();

    let args = Args::parse();
    let input_file = args.input;
    log::debug!("Interpretting {input_file:?}");
    let mut f = File::open(input_file)?;
    let mut contents = String::new();
    f.read_to_string(&mut contents)?;
    let p = Program::new(&contents);
    let mut p = match p {
        Ok(p) => p,
        Err(e) => {
            let mut s = String::new();
            GraphicalReportHandler::new()
                .render_report(&mut s, &e)
                .unwrap();
            bail!("{s}");
        }
    };
    histogram();
    cumulative_histogram();
    log::info!("{:?}", p.run());
    Ok(())
}
