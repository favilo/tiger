use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take_until},
    character::complete::{alpha1, char, digit1, multispace0, multispace1},
    combinator::all_consuming,
    multi::{many0, many1},
    sequence::{preceded, terminated, tuple},
};

#[derive(Default, Debug)]
pub struct ParseError;

#[derive(Default, Debug)]
pub struct Program {}

#[derive(Debug)]
enum Op {
    Plus,
    Minus,
    Times,
    Divide,
}

#[derive(Debug)]
enum Expression {
    Id(String),
    Nil,
    IntLit(i64),
    StringLit(String),

    Call {
        name: String,
        parameters: Vec<Expression>,
    },

    InfixOp {
        op_type: Op,
        first: Box<Expression>,
        second: Box<Expression>,
    },
}

impl std::error::Error for ParseError {}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A parsing error occurred.")
    }
}

pub fn parse_program(input: &str) -> Result<Program, ParseError> {
    let output = program(input);
    log::info!("{:?}", output);
    Ok(Program::default())
}

fn program(input: &str) -> nom::IResult<&str, Expression> {
    all_consuming(exp)(input)
}

fn exp(input: &str) -> nom::IResult<&str, Expression> {
    surrounded(
        whitespace,
        alt((call_exp, nil_exp, string_exp, int_lit_exp, id_exp)),
        whitespace,
    )(input)
}

fn id_exp(input: &str) -> nom::IResult<&str, Expression> {
    let (input, name) = id(input)?;
    Ok((input, Expression::Id(name.to_string())))
}

fn nil_exp(input: &str) -> nom::IResult<&str, Expression> {
    let (input, _) = tag("nil")(input)?;
    Ok((input, Expression::Nil))
}

fn string_exp(input: &str) -> nom::IResult<&str, Expression> {
    let (input, (_, s, _)) = tuple((tag("\""), is_not("\""), tag("\"")))(input)?;
    Ok((input, Expression::StringLit(s.to_string())))
}

fn int_lit_exp(input: &str) -> nom::IResult<&str, Expression> {
    let (input, i) = digit1(input)?;
    Ok((
        input,
        Expression::IntLit(i.parse().expect("Only digits should be possible")),
    ))
}

fn call_exp(input: &str) -> nom::IResult<&str, Expression> {
    let (input, (name, _, params, _)) = tuple((id, tag("("), exp_list_comma, tag(")")))(input)?;
    Ok((
        input,
        Expression::Call {
            name: name.to_string(),
            parameters: params,
        },
    ))
}

fn exp_list_comma(input: &str) -> nom::IResult<&str, Vec<Expression>> {
    let (input, (mut head, mut last)) = tuple((many0(terminated(exp, tag(","))), exp_as_list))(input)?;
    head.append(&mut last);
    Ok((input, head))
}

fn exp_as_list(input: &str) -> nom::IResult<&str, Vec<Expression>> {
    let (input, e) = exp(input)?;
    Ok((input, vec![e]))
}

// TODO: Fix this, inelligent
fn id(input: &str) -> nom::IResult<&str, String> {
    let (input, (head, rest)) = tuple((alpha1, many0(alt((alpha1, digit1, tag("_"))))))(input)?;
    let mut s = String::new();
    s += head;
    s += &rest.join("");
    Ok((input, s))
}

fn whitespace(input: &str) -> nom::IResult<&str, &str> {
    let (input, _) = many0(alt((multispace1, comment)))(input)?;
    Ok((input, ""))
}

// TODO: Implement nested comments
fn comment(input: &str) -> nom::IResult<&str, &str> {
    let (input, _) = surrounded(tag("/*"), take_until("*/"), tag("*/"))(input)?;
    Ok((input, ""))
}

fn surrounded<I, O1, O2, O3, E: nom::error::ParseError<I>, F, G, H>(
    open: F,
    mid: G,
    close: H,
) -> impl Fn(I) -> nom::IResult<I, O2, E>
where
    F: Fn(I) -> nom::IResult<I, O1, E>,
    G: Fn(I) -> nom::IResult<I, O2, E>,
    H: Fn(I) -> nom::IResult<I, O3, E>,
{
    terminated(preceded(open, mid), close)
}
