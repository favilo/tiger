use miette::Diagnostic;
use nom_supreme::error::{BaseErrorKind, GenericErrorTree, StackContext};
use thiserror::Error;

use crate::lang::NomParseError;

#[derive(Error, Debug, Diagnostic)]
#[error("Parse Error")]
pub struct FormattedError<'input> {
    #[source_code]
    src: &'input str,

    #[label("{kind}")]
    span: miette::SourceSpan,

    kind: BaseErrorKind<&'input str, Box<dyn std::error::Error + Send + Sync + 'static>>,

    #[related]
    others: Vec<FormattedErrorContext<'input>>,
}

impl<'input> FormattedError<'input> {
    pub(crate) fn from_nom(input: &'input str, e: NomParseError<'input>) -> Self {
        match e {
            GenericErrorTree::Base { location, kind } => {
                let offset = location.location_offset().into();
                FormattedError {
                    src: input,
                    span: miette::SourceSpan::new(offset, 0.into()),
                    kind,
                    others: vec![],
                }
            }
            GenericErrorTree::Stack { base, contexts } => {
                let mut base = FormattedError::from_nom(input, *base);
                let contexts = contexts
                    .into_iter()
                    .map(|(location, context)| {
                        let offset = location.location_offset().into();
                        FormattedErrorContext {
                            src: input,
                            span: miette::SourceSpan::new(offset, 0.into()),
                            context,
                        }
                    })
                    .collect::<Vec<_>>();

                base.others.extend(contexts);
                base
            }
            GenericErrorTree::Alt(alt_errors) => alt_errors
                .into_iter()
                .map(|e| FormattedError::from_nom(input, e))
                .max_by_key(|formatted| formatted.others.len())
                .unwrap(),
        }
    }
}

#[derive(Error, Debug, Diagnostic)]
#[error("Parse Error context")]
pub struct FormattedErrorContext<'input> {
    #[source_code]
    src: &'input str,

    #[label("{context}")]
    span: miette::SourceSpan,

    context: StackContext<&'input str>,
}

#[derive(Error, Debug, Diagnostic)]
pub enum Error<'input> {
    #[error("Parse Error")]
    #[diagnostic(code(lang::parse_error))]
    ParseError(NomParseError<'input>),

    #[error("Interpretter Error")]
    #[diagnostic(code(eval::interpret_error))]
    InterpretError(#[from] InterpretterError),
}

impl<'input> From<NomParseError<'input>> for Error<'input> {
    fn from(value: NomParseError<'input>) -> Self {
        Error::ParseError(value)
    }
}

#[derive(Error, Debug, Diagnostic)]
pub enum InterpretterError {
    #[error("Type Error: {0}")]
    #[diagnostic(code(interpret::types))]
    TypeError(String),
}
