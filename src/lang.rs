use std::{collections::BTreeMap, convert::TryFrom};

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{line_ending, multispace0},
    combinator::{all_consuming, map, opt},
    multi::{fold_many0, many1},
    sequence::{delimited, tuple},
    Finish, IResult, Parser,
};
use nom_locate::LocatedSpan;
use nom_recursive::{recursive_parser, RecursiveInfo};
use nom_supreme::{error::ErrorTree, ParserExt};

use crate::error::{Error, FormattedError, InterpretterError};

pub trait Parse<'input>: Sized {
    /// Parse the given span into self
    fn parse(input: Span<'input>) -> ParseResult<'input, Self>;

    /// Helper method for tests to convert a str into a raw span and parse
    fn parse_from_raw(input: &'input str) -> ParseResult<'input, Self> {
        let i = LocatedSpan::new_extra(input, RecursiveInfo::new());
        Self::parse(i)
    }
}

#[derive(Default, Debug)]
pub struct Program {
    expressions: Vec<Expression>,
    _variables: BTreeMap<String, Expression>,
    // TODO: Decide how to define function definitions.
    // It'll probably be a new struct
    // _functions: BTreeMap<String, Expression>,
}

impl Program {
    pub fn new(input: &str) -> Result<Self, FormattedError> {
        let i = LocatedSpan::new_extra(input, RecursiveInfo::new());
        match all_consuming(Self::parse)(i).finish() {
            Ok((_, this)) => Ok(this),
            Err(e) => Err(FormattedError::from_nom(input, e)),
        }
    }

    fn eval(&mut self, expr: &Expression) -> Result<Expression, InterpretterError> {
        Ok(match expr {
            Expression::IntLit(_) | Expression::StringLit(_) | Expression::Nil => expr.clone(),
            Expression::BinaryOp {
                op_type,
                first,
                second,
            } => {
                let first = self.eval(first)?.as_int()?;
                let second = self.eval(second)?.as_int()?;
                match op_type {
                    Op::Plus => Expression::IntLit(first + second),
                    Op::Minus => Expression::IntLit(first - second),
                    Op::Times => Expression::IntLit(first * second),
                    Op::Divide => Expression::IntLit(first / second),

                    Op::Lt => Expression::IntLit((first < second) as i64),
                    Op::Gt => Expression::IntLit((first > second) as i64),
                    Op::LtEqual => Expression::IntLit((first >= second) as i64),
                    Op::GtEqual => Expression::IntLit((first >= second) as i64),
                    Op::Equal => Expression::IntLit((first == second) as i64),
                    Op::NotEqual => Expression::IntLit((first != second) as i64),

                    Op::And => Expression::IntLit(((first != 0) && (second != 0)) as i64),
                    Op::Or => Expression::IntLit(((first != 0) || (second != 0)) as i64),
                }
            }
            Expression::UnaryOp(Op::Minus, expr) => Expression::IntLit(-self.eval(expr)?.as_int()?),
            Expression::UnaryOp(_, _) => unreachable!("Unary isn't defined for any other op"),
            Expression::Id(_name) => todo!(),
            Expression::Assign(_name, _expr) => todo!(),
            Expression::Call {
                name: _,
                parameters: _,
            } => todo!(),
            Expression::Block(exprs) => exprs
                .iter()
                .map(|expr| self.eval(expr))
                .last()
                .unwrap_or(Ok(Expression::Nil))?,
        })
    }

    pub fn run(&mut self) -> Result<Expression, InterpretterError> {
        let mut result = Expression::IntLit(0);
        let expressions = self.expressions.clone();
        for expr in &expressions {
            result = self.eval(expr)?;
        }
        Ok(result)
    }
}

impl<'input> Parse<'input> for Program {
    fn parse(input: Span<'input>) -> ParseResult<'input, Self> {
        let exprs = many1(Expression::parse.terminated(tuple((
            multispace0,
            opt(tag(";").delimited_by(multispace0)),
            opt(line_ending),
        ))));
        map(exprs, |expressions| Self {
            expressions,
            ..Default::default()
        })(input)
    }
}

pub type Span<'input> = LocatedSpan<&'input str, RecursiveInfo>;
pub type NomParseError<'input> = ErrorTree<Span<'input>>;
pub type ParseResult<'input, T> = IResult<Span<'input>, T, NomParseError<'input>>;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Op {
    Plus,
    Minus,
    Times,
    Divide,

    Lt,
    Gt,
    LtEqual,
    GtEqual,
    Equal,
    NotEqual,

    And,
    Or,
}

impl Op {
    pub fn from_input(input: Span<'_>) -> Result<Self, Error> {
        Ok(match *input.fragment() {
            "*" => Op::Times,
            "/" => Op::Divide,
            "+" => Op::Plus,
            "-" => Op::Minus,

            ">" => Op::Gt,
            "<" => Op::Lt,
            ">=" => Op::GtEqual,
            "<=" => Op::LtEqual,
            "=" => Op::Equal,
            "<>" => Op::NotEqual,

            "&" => Op::And,
            "|" => Op::Or,
            c => unreachable!("Didn't address binary op {}", c),
        })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    Nil,
    IntLit(i64),
    StringLit(String),

    Id(String),
    Assign(String, Box<Expression>),
    Call {
        name: String,
        parameters: Vec<Expression>,
    },

    BinaryOp {
        op_type: Op,
        first: Box<Expression>,
        second: Box<Expression>,
    },

    UnaryOp(Op, Box<Expression>),

    Block(Vec<Expression>),
}

impl<'s> Parse<'s> for Expression {
    #[recursive_parser]
    fn parse(s: Span<'s>) -> ParseResult<'s, Self> {
        Self::expr(s)
    }
}

impl Expression {
    pub fn as_int(&self) -> Result<i64, InterpretterError> {
        match self {
            Expression::IntLit(i) => Ok(*i),
            e => Err(InterpretterError::TypeError(format!(
                "{e:?} is not an integer"
            ))),
        }
    }

    fn int_lit(input: Span<'_>) -> ParseResult<'_, Self> {
        map(
            nom::character::complete::u64.delimited_by(multispace0),
            |int| Expression::IntLit(i64::try_from(int).expect("TODO: Fix parsing ints")),
        )
        .context("Integer Literal")
        .parse(input)
    }

    fn string_lit(input: Span<'_>) -> ParseResult<'_, Self> {
        // TODO: Deal with escaping
        map(
            delimited(tag("\""), take_until("\""), tag("\"")),
            |s: LocatedSpan<&str, _>| Expression::StringLit(s.fragment().to_string()),
        )
        .context("String Literal")
        .parse(input)
    }

    fn nil_lit(input: Span<'_>) -> ParseResult<'_, Self> {
        map(tag("nil"), |_| Expression::Nil)
            .delimited_by(multispace0)
            .context("Nil Literal")
            .parse(input)
    }

    fn lit(input: Span<'_>) -> ParseResult<'_, Self> {
        alt((Self::int_lit, Self::string_lit, Self::nil_lit))
            .delimited_by(multispace0)
            .context("Literal")
            .parse(input)
    }

    #[recursive_parser]
    fn atom(s: Span<'_>) -> ParseResult<'_, Self> {
        alt((
            Self::lit.context("literal"),
            delimited(
                tag("(").delimited_by(multispace0),
                Self::exprs,
                tag(")").delimited_by(multispace0),
            )
            .context("parentheses"),
        ))
        .context("atom")
        .parse(s)
    }

    fn unary(input: Span<'_>) -> ParseResult<'_, Self> {
        map(
            tuple((
                fold_many0(
                    tag("-").delimited_by(multispace0),
                    Vec::new,
                    |mut acc: Vec<_>, item| {
                        acc.push(item);
                        acc
                    },
                ),
                Self::atom,
            )),
            |(minuses, atom)| {
                let expr = minuses.iter().fold(atom, |expr, _minus| {
                    Expression::UnaryOp(Op::Minus, Box::new(expr))
                });
                expr
            },
        )(input)
    }

    fn product(input: Span<'_>) -> ParseResult<'_, Self> {
        let (input, lhs) = Self::unary.context("lhs operand").parse(input)?;
        let (input, rest) = fold_many0(
            tuple((
                alt((tag("*"), tag("/")))
                    .context("product")
                    .delimited_by(multispace0),
                Self::unary.context("rhs operand"),
            )),
            Vec::new,
            |mut acc: Vec<_>, (op, operand)| {
                let op = Op::from_input(op).expect("Op should be completely defined");
                acc.push((op, operand));
                acc
            },
        )(input)?;
        let lhs = rest
            .into_iter()
            .fold(lhs, |lhs, (op, rhs)| Expression::BinaryOp {
                op_type: op,
                first: Box::new(lhs),
                second: Box::new(rhs),
            });
        Ok((input, lhs))
    }

    fn sum(input: Span<'_>) -> ParseResult<'_, Self> {
        let (input, lhs) = Self::product.context("lhs operand").parse(input)?;
        let (input, rest) = fold_many0(
            tuple((
                alt((tag("+"), tag("-")))
                    .context("sum")
                    .delimited_by(multispace0),
                Self::product.context("rhs operand"),
            )),
            Vec::new,
            |mut acc: Vec<_>, (op, operand)| {
                let op = Op::from_input(op).expect("Op should be completely defined");
                acc.push((op, operand));
                acc
            },
        )(input)?;
        let lhs = rest
            .into_iter()
            .fold(lhs, |lhs, (op, rhs)| Expression::BinaryOp {
                op_type: op,
                first: Box::new(lhs),
                second: Box::new(rhs),
            });
        Ok((input, lhs))
    }

    fn cmp(input: Span<'_>) -> ParseResult<'_, Self> {
        let (input, lhs) = Self::sum.context("lhs operand").parse(input)?;
        let (input, rest) = fold_many0(
            tuple((
                alt((
                    tag(">=").context("greater than or equal"),
                    tag("<=").context("less than or equal"),
                    tag("=").context("equals"),
                    tag("<>").context("not equals"),
                    tag(">").context("greater than"),
                    tag("<").context("less than"),
                ))
                .context("comparison op")
                .delimited_by(multispace0),
                Self::sum.context("rhs operand"),
            )),
            Vec::new,
            |mut acc: Vec<_>, (op, operand)| {
                let op = Op::from_input(op).expect("Op should be completely defined");
                acc.push((op, operand));
                acc
            },
        )(input)?;
        let lhs = rest
            .into_iter()
            .fold(lhs, |lhs, (op, rhs)| Expression::BinaryOp {
                op_type: op,
                first: Box::new(lhs),
                second: Box::new(rhs),
            });
        Ok((input, lhs))
    }

    fn and(input: Span<'_>) -> ParseResult<'_, Self> {
        let (input, lhs) = Self::cmp.context("lhs operand").parse(input)?;
        let (input, rest) = fold_many0(
            tuple((
                alt((tag("&").context("and"),))
                    .context("boolean and")
                    .delimited_by(multispace0),
                Self::cmp.context("rhs operand"),
            )),
            Vec::new,
            |mut acc: Vec<_>, (op, operand)| {
                let op = Op::from_input(op).expect("Op should be completely defined");
                acc.push((op, operand));
                acc
            },
        )(input)?;
        let lhs = rest
            .into_iter()
            .fold(lhs, |lhs, (op, rhs)| Expression::BinaryOp {
                op_type: op,
                first: Box::new(lhs),
                second: Box::new(rhs),
            });
        Ok((input, lhs))
    }

    fn or(input: Span<'_>) -> ParseResult<'_, Self> {
        let (input, lhs) = Self::and.context("lhs operand").parse(input)?;
        let (input, rest) = fold_many0(
            tuple((
                alt((tag("|").context("and"),))
                    .context("boolean or")
                    .delimited_by(multispace0),
                Self::and.context("rhs operand"),
            )),
            Vec::new,
            |mut acc: Vec<_>, (op, operand)| {
                let op = Op::from_input(op).expect("Op should be completely defined");
                acc.push((op, operand));
                acc
            },
        )(input)?;
        let lhs = rest
            .into_iter()
            .fold(lhs, |lhs, (op, rhs)| Expression::BinaryOp {
                op_type: op,
                first: Box::new(lhs),
                second: Box::new(rhs),
            });
        Ok((input, lhs))
    }

    #[recursive_parser]
    fn expr(s: Span) -> ParseResult<Self> {
        Self::or(s)
    }

    #[recursive_parser]
    fn exprs(s: Span) -> ParseResult<Self> {
        map(
            many1(Self::expr.terminated(opt(tag(";").delimited_by(multispace0)))),
            |v| Self::Block(v),
        )(s)
    }
}

#[cfg(test)]
mod tests {
    use miette::GraphicalReportHandler;

    use super::*;

    #[test]
    fn nil() {
        let input = "nil";
        let e = Program::new(input).unwrap().run().unwrap();
        assert_eq!(e, Expression::Nil);
    }

    #[test]
    fn int() {
        let input = "10";
        let e = Program::new(input).unwrap().run().unwrap();
        assert_eq!(e, Expression::IntLit(10));
    }

    #[test]
    fn int_spaces() {
        let input = " 10 ";
        let e = Program::new(input).unwrap().run().unwrap();
        assert_eq!(e, Expression::IntLit(10));
    }

    #[test]
    fn int_neg() {
        let input = "-10";
        let mut p = Program::new(input).unwrap();
        let e = &p.expressions[0];
        assert_eq!(
            e,
            &Expression::UnaryOp(Op::Minus, Box::new(Expression::IntLit(10)))
        );
        let e = p.run().unwrap();
        assert_eq!(e, Expression::IntLit(-10));
    }

    #[test]
    fn int_endl() {
        let input = "10\n";
        let e = Program::new(input).unwrap().run().unwrap();
        assert_eq!(e, Expression::IntLit(10));
    }

    #[test]
    fn int_semi() {
        let input = "10;";
        let e = Program::new(input).unwrap().run().unwrap();
        assert_eq!(e, Expression::IntLit(10));
    }

    #[test]
    fn simple_string_lit() {
        let input = r#""test""#;
        let e = &Program::new(input).unwrap().run().unwrap();
        assert_eq!(e, &Expression::StringLit("test".into()));
    }

    // TODO: Come back to this
    #[test]
    #[ignore]
    fn escaped_string_lit() {
        let input = r#""\"test""#;
        let e = &Program::new(input).unwrap().run().unwrap();
        assert_eq!(e, &Expression::StringLit("\"test".into()));
    }

    #[test]
    #[ignore]
    fn test_hello_world() {
        let input = " /* Comment */\n\
                     print(\"Hello, World!\\n\", \"Testing\", 1, 2, nil)\n";
        let p = Program::new(&input).unwrap();
        assert_eq!(input, "");
        assert_eq!(
            p.expressions,
            vec![Expression::Call {
                name: "print".to_string(),
                parameters: vec![
                    Expression::StringLit("Hello, World!\\n".to_string()),
                    Expression::StringLit("Testing".to_string()),
                    Expression::IntLit(1),
                    Expression::IntLit(2),
                    Expression::Nil
                ]
            }],
        );
    }

    #[test]
    fn product() {
        let input = "5 * 4";
        let mut p = Program::new(input).unwrap();
        assert_eq!(
            p.expressions[0],
            Expression::BinaryOp {
                op_type: Op::Times,
                first: Box::new(Expression::IntLit(5)),
                second: Box::new(Expression::IntLit(4))
            }
        );
        assert_eq!(p.run().unwrap().as_int().unwrap(), 20);
    }

    #[test]
    fn product_multi() {
        let input = "6 * 3 / 2";
        let mut p = Program::new(input).unwrap();
        assert_eq!(
            p.expressions[0],
            Expression::BinaryOp {
                op_type: Op::Divide,
                first: Box::new(Expression::BinaryOp {
                    op_type: Op::Times,
                    first: Box::new(Expression::IntLit(6)),
                    second: Box::new(Expression::IntLit(3))
                }),
                second: Box::new(Expression::IntLit(2)),
            }
        );
        assert_eq!(p.run().unwrap().as_int().unwrap(), 9);
    }

    #[test]
    fn sum_multi() {
        let input = "6 * 3 - 2";
        let mut p = Program::new(input).unwrap();
        assert_eq!(
            p.expressions[0],
            Expression::BinaryOp {
                op_type: Op::Minus,
                first: Box::new(Expression::BinaryOp {
                    op_type: Op::Times,
                    first: Box::new(Expression::IntLit(6)),
                    second: Box::new(Expression::IntLit(3))
                }),
                second: Box::new(Expression::IntLit(2)),
            }
        );
        assert_eq!(p.run().unwrap().as_int().unwrap(), 16);
    }

    #[test]
    fn sum_backward() {
        let input = "2 - 6 * -3 ";
        let mut p = Program::new(input).unwrap();
        assert_eq!(
            p.expressions[0],
            Expression::BinaryOp {
                op_type: Op::Minus,
                first: Box::new(Expression::IntLit(2)),
                second: Box::new(Expression::BinaryOp {
                    op_type: Op::Times,
                    first: Box::new(Expression::IntLit(6)),
                    second: Box::new(Expression::UnaryOp(
                        Op::Minus,
                        Box::new(Expression::IntLit(3))
                    )),
                }),
            }
        );
        assert_eq!(p.run().unwrap().as_int().unwrap(), 20);
    }

    #[test]
    fn parentheses() {
        let input = " 3 * (4 + 2) ";
        let mut p = Program::new(input).unwrap();
        assert_eq!(p.run().unwrap().as_int().unwrap(), 18);
        assert_eq!(
            p.expressions[0],
            Expression::BinaryOp {
                op_type: Op::Times,
                first: Box::new(Expression::IntLit(3)),
                // Block because of weird parentheses rules
                second: Box::new(Expression::Block(vec![Expression::BinaryOp {
                    op_type: Op::Plus,
                    first: Box::new(Expression::IntLit(4)),
                    second: Box::new(Expression::IntLit(2)),
                }])),
            }
        );
    }

    #[test]
    fn comparisons() {
        let input = " 3 > 5 ";
        let mut p = Program::new(input).unwrap();
        assert_eq!(p.run().unwrap().as_int().unwrap(), 0);
        assert_eq!(
            p.expressions[0],
            Expression::BinaryOp {
                op_type: Op::Gt,
                first: Box::new(Expression::IntLit(3)),
                second: Box::new(Expression::IntLit(5)),
            }
        );
    }

    #[test]
    fn boolean_ops() {
        let input = " (3 > 5 | 5 < 10) & 5 <>3";
        let p = Program::new(input);
        let mut p = match p {
            Ok(p) => p,
            Err(e) => {
                let mut s = String::new();
                GraphicalReportHandler::new()
                    .render_report(&mut s, &e)
                    .unwrap();
                panic!("{}", s);
            }
        };
        assert_eq!(p.run().unwrap().as_int().unwrap(), 1);
    }
}
