use std::{collections::BTreeMap, convert::TryFrom};

use miette::GraphicalReportHandler;

#[allow(unused_imports)]
use nom::{
    branch::alt,
    bytes::complete::{tag, take_until, take_while},
    character::{
        complete::{alpha1, multispace0, multispace1},
        is_alphanumeric,
    },
    combinator::{all_consuming, fail, map, not, opt, recognize},
    multi::{fold_many0, many0, many1},
    sequence::{delimited, tuple},
    Finish, IResult, Parser,
};

use nom_locate::LocatedSpan;
use nom_recursive::{recursive_parser, HasRecursiveInfo, RecursiveInfo};
use nom_supreme::{error::ErrorTree, ParserExt};
use nom_tracable::{tracable_parser, HasTracableInfo, TracableInfo};

use crate::error::{Error, FormattedError, InterpretterError};

pub trait Parse<'input>: Sized {
    /// Parse the given span into self
    fn parse(input: Span<'input>) -> ParseResult<'input, Self>;

    /// Helper method for tests to convert a str into a raw span and parse
    fn parse_from_raw(input: &'input str) -> ParseResult<'input, Self> {
        let i = Span::new_extra(input, ParserInfo::default());
        Self::parse(i)
    }
}

trait PrettyUnwrap<T>: Sized {
    fn pretty_unwrap(self) -> T;
}

impl<T> PrettyUnwrap<T> for Result<T, FormattedError<'_>> {
    fn pretty_unwrap(self) -> T {
        match self {
            Ok(p) => p,
            Err(e) => {
                let mut s = String::new();
                GraphicalReportHandler::new()
                    .render_report(&mut s, &e)
                    .unwrap();
                panic!("{}", s);
            }
        }
    }
}

#[derive(Default, Debug)]
pub struct Program {
    /// The expressions that make up the program
    // TODO: Make this A Source type instead of Vec<Expression>
    expression: Expression,

    /// Stack of variable definitions
    variables: Vec<BTreeMap<String, Expression>>,
    // TODO: Decide how to define function definitions.
    // It'll probably be a new struct
    // _functions: BTreeMap<String, FnDec>,

    // TODO: Types need to happen too.
    // _types: BTreeMap<String, TypeDec>
}

impl Program {
    pub fn new(input: &str) -> Result<Self, FormattedError> {
        let i = Span::new_extra(input, ParserInfo::default());
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
            Expression::Id(name) => self
                .variables
                .last()
                .ok_or(InterpretterError::NameError(name.to_string()))?
                .get(name)
                .ok_or(InterpretterError::NameError(name.to_string()))?
                .clone(),
            Expression::Assign(name, expr) => {
                if !self
                    .variables
                    .last()
                    .ok_or(InterpretterError::NameError(name.to_string()))?
                    .contains_key(name)
                {
                    Err(InterpretterError::NameError(name.to_string()))?;
                }
                let value = self.eval(expr)?;
                self.variables[0].insert(name.clone(), value);
                Expression::Nil
            }
            Expression::Block(exprs) => exprs
                .iter()
                .map(|expr| self.eval(expr))
                .last()
                .unwrap_or(Ok(Expression::Nil))?,
            Expression::Let { decls, exprs } => {
                // Push new stack
                let mut vars = BTreeMap::default();
                decls.into_iter().try_for_each(|d| {
                    d.assign_value(self, &mut vars)?;
                    Ok(())
                })?;

                self.variables.push(vars);
                // evaluate exprs
                let ret = self.eval(exprs)?;

                // Pop stack
                self.variables.pop();
                ret
            }
            Expression::Call {
                name: _,
                parameters: _,
            } => todo!(),
        })
    }

    pub fn run(&mut self) -> Result<Expression, InterpretterError> {
        let expr = self.expression.clone();
        let result = self.eval(&expr)?;
        Ok(result)
    }
}

impl<'input> Parse<'input> for Program {
    fn parse(input: Span<'input>) -> ParseResult<'input, Self> {
        let expr = Expression::parse;
        map(expr, |expression| Self {
            expression,
            ..Default::default()
        })(input)
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct ParserInfo {
    recursive: RecursiveInfo,
    tracable: TracableInfo,
}

impl Default for ParserInfo {
    fn default() -> Self {
        Self {
            tracable: TracableInfo::default()
                .forward(true)
                .backward(true)
                .parser_width(64)
                .fragment_width(20)
                .color(true),
            recursive: RecursiveInfo::default(),
        }
    }
}

impl HasRecursiveInfo for ParserInfo {
    fn get_recursive_info(&self) -> RecursiveInfo {
        self.recursive
    }

    fn set_recursive_info(mut self, info: RecursiveInfo) -> Self {
        self.recursive = info;
        self
    }
}

impl HasTracableInfo for ParserInfo {
    fn get_tracable_info(&self) -> TracableInfo {
        self.tracable
    }

    fn set_tracable_info(mut self, info: TracableInfo) -> Self {
        self.tracable = info;
        self
    }
}

pub type Span<'input> = LocatedSpan<&'input str, ParserInfo>;
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

pub type TypeId = String;

#[derive(Debug, PartialEq, Clone)]
pub enum Declaration {
    // Type
    // Class
    Variable {
        id: String,
        type_id: Option<TypeId>,
        expr: Expression,
    },
    // Function {
    //     id: String,
    //     dec: TyFields,
    //     ret: TypeId ,
    //     expr: Expression
    // },
    // Primitive {
    //     id: String,
    //     dec: TyFields,
    //     ret: TypeId ,
    // },
    // Import(String),
}

impl Parse<'_> for Declaration {
    #[tracable_parser]
    fn parse(input: Span) -> ParseResult<Self> {
        alt((Self::vardec.context("variable declaration"),))(input)
    }
}

impl Declaration {
    pub fn assign_value(
        &self,
        p: &mut Program,
        vars: &mut BTreeMap<String, Expression>,
    ) -> Result<(), InterpretterError> {
        match self {
            Declaration::Variable {
                id,
                type_id: _,
                expr,
            } => {
                vars.insert(id.clone(), p.eval(expr)?);
            }
        }
        Ok(())
    }

    #[tracable_parser]
    fn var_id(input: Span) -> ParseResult<String> {
        id.preceded_by(tag("var").context("var keyword").delimited_by(multispace0))
            .context("variable id")
            .parse(input)
    }

    #[tracable_parser]
    fn var_type(input: Span) -> ParseResult<Option<String>> {
        opt(id
            .preceded_by(tag(":").cut().context("colon").delimited_by(multispace0))
            .context("variable type"))
        .parse(input)
    }

    fn var_value(input: Span) -> ParseResult<Expression> {
        Expression::expr
            .context("variable value")
            .preceded_by(
                tag(":=")
                    .context("assignment operator")
                    .delimited_by(multispace0),
            )
            .parse(input)
    }

    #[tracable_parser]
    fn vardec(input: Span) -> ParseResult<Self> {
        let (input, var_id) = Self::var_id.parse(input)?;

        let (input, type_id) = Self::var_type.parse(input)?;

        let (input, expr) = Self::var_value.parse(input)?;

        Ok((
            input,
            Declaration::Variable {
                id: var_id,
                type_id,
                expr,
            },
        ))
    }
}

#[derive(Default, Debug, PartialEq, Clone)]
pub enum Expression {
    #[default]
    Nil,
    IntLit(i64),
    StringLit(String),

    Id(String),
    Assign(String, Box<Expression>),
    Let {
        decls: Vec<Declaration>,
        exprs: Box<Expression>,
    },

    #[allow(dead_code)]
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

// TODO: Comments
impl Expression {
    pub fn as_int(&self) -> Result<i64, InterpretterError> {
        match self {
            Expression::IntLit(i) => Ok(*i),
            e => Err(InterpretterError::TypeError(format!(
                "{e:?} is not an integer"
            ))),
        }
    }

    #[tracable_parser]
    fn int_lit(input: Span<'_>) -> ParseResult<'_, Self> {
        map(
            nom::character::complete::u64.delimited_by(multispace0),
            |int| Expression::IntLit(i64::try_from(int).expect("TODO: Fix parsing ints")),
        )
        .context("Integer Literal")
        .parse(input)
    }

    #[tracable_parser]
    fn string_lit(input: Span<'_>) -> ParseResult<'_, Self> {
        // TODO: Deal with escaping
        map(
            delimited(tag("\""), take_until("\""), tag("\"")),
            |s: Span| Expression::StringLit(s.fragment().to_string()),
        )
        .context("String Literal")
        .parse(input)
    }

    #[tracable_parser]
    fn nil_lit(input: Span<'_>) -> ParseResult<'_, Self> {
        map(tag("nil"), |_| Expression::Nil)
            .delimited_by(multispace0)
            .context("Nil Literal")
            .parse(input)
    }

    #[tracable_parser]
    fn lit(input: Span<'_>) -> ParseResult<'_, Self> {
        alt((Self::int_lit, Self::string_lit, Self::nil_lit))
            .delimited_by(multispace0)
            .parse(input)
    }

    #[tracable_parser]
    fn lvalue(input: Span) -> ParseResult<Self> {
        map(id, |name| Expression::Id(name)).parse(input)
    }

    #[tracable_parser]
    fn assign(input: Span) -> ParseResult<Self> {
        map(
            tuple((
                Self::lvalue.terminated(tag(":=").delimited_by(multispace0)),
                Self::expr,
            )),
            |(name, expr)| {
                let Expression::Id(name) = name else {panic!("Not sure how we got here")};
                Expression::Assign(name, Box::new(expr))
            },
        )
        .parse(input)
    }

    #[recursive_parser]
    #[tracable_parser]
    fn atom(s: Span<'_>) -> ParseResult<'_, Self> {
        alt((
            delimited(
                tag("(").delimited_by(multispace0),
                Self::exprs.context("exprs"),
                tag(")").delimited_by(multispace0),
            )
            .context("parentheses"),
            Self::lit.context("literal"),
            Self::assign.context("assignment"),
            Self::lvalue.context("lvalue"),
        ))
        .context("atom")
        .parse(s)
    }

    #[tracable_parser]
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

    #[tracable_parser]
    fn product(input: Span<'_>) -> ParseResult<'_, Self> {
        let (input, lhs) = Self::unary.context("product lhs operand").parse(input)?;
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

    #[tracable_parser]
    fn sum(input: Span<'_>) -> ParseResult<'_, Self> {
        let (input, lhs) = Self::product.context("sum lhs operand").parse(input)?;
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

    #[tracable_parser]
    fn cmp(input: Span<'_>) -> ParseResult<'_, Self> {
        let (input, lhs) = Self::sum.context("cmp lhs operand").parse(input)?;
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

    #[tracable_parser]
    fn and(input: Span<'_>) -> ParseResult<'_, Self> {
        let (input, lhs) = Self::cmp.context("and lhs operand").parse(input)?;
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

    #[tracable_parser]
    fn or(input: Span<'_>) -> ParseResult<'_, Self> {
        let (input, lhs) = Self::and.context("or lhs operand").parse(input)?;
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
    #[tracable_parser]
    fn let_expr(s: Span) -> ParseResult<Self> {
        let (input, decls) = many0(Declaration::parse)
            .preceded_by(tag("let").delimited_by(multispace0))
            .parse(s)?;
        let (input, exprs) = Self::exprs
            .context("in block")
            .preceded_by(tag("in").context("in keyword").delimited_by(multispace0))
            .terminated(tag("end").context("end keyword").delimited_by(multispace0))
            .parse(input)?;
        Ok((
            input,
            Expression::Let {
                decls,
                exprs: Box::new(exprs),
            },
        ))
    }

    #[recursive_parser]
    #[tracable_parser]
    fn expr(s: Span) -> ParseResult<Self> {
        let (s, ()) = not(tag("end"))(s)?;
        alt((
            Self::let_expr.context("let expression"),
            Self::or.context("normal expression"),
        ))
        .parse(s)
    }

    #[recursive_parser]
    #[tracable_parser]
    fn exprs(s: Span) -> ParseResult<Self> {
        map(
            many1(
                Self::expr
                    .context("exprs")
                    .terminated(opt(tag(";").delimited_by(multispace0))),
            ),
            |v| Self::Block(v),
        )(s)
    }
}

#[tracable_parser]
pub(crate) fn any_keyword(input: Span) -> ParseResult<String> {
    // Normal keywords
    let parse = alt((
        tag("array").context("array keyword"),
        tag("if").context("if keyword"),
        tag("then").context("then keyword"),
        tag("else").context("else keyword"),
        tag("while").context("while keyword"),
        tag("for").context("for keyword"),
        tag("to").context("to keyword"),
        tag("do").context("do keyword"),
        tag("let").context("let keyword"),
        tag("in").context("in keyword"),
        tag("end").context("end keyword"),
        tag("of").context("of keyword"),
        tag("break").context("break keyword"),
        tag("nil").context("nil keyword"),
        tag("function").context("function keyword"),
        tag("var").context("var keyword"),
        tag("type").context("type keyword"),
        tag("import").context("import keyword"),
        tag("primitive").context("primitive keyword"),
    ))
    // Object stuff
    .or(alt((
        tag("object").context("object keyword"),
        tag("extends").context("extends keyword"),
        tag("method").context("method keyword"),
        tag("new").context("new keyword"),
    )))
    .delimited_by(multispace1)
    .map(|keyword: Span| keyword.to_string())
    .parse(input);
    if input.fragment() == &"end" {
        panic!("end found {:?}", parse);
    }
    parse
}

#[tracable_parser]
pub(crate) fn id(input: Span) -> ParseResult<String> {
    let (input, ()) = not(any_keyword).context("not keyword").parse(input)?;
    let (input, id) = alt((
        tag("_main").context("_main keyword"),
        recognize(tuple((
            alpha1::<Span, _>.context("head"),
            take_while(|c| is_alphanumeric(c as u8) || c == '_'),
        ))),
    ))
    .delimited_by(multispace0)
    .map(|name| name.to_string())
    .parse(input)?;
    Ok((input, id))
}

#[cfg(test)]
mod tests {
    use miette::GraphicalReportHandler;
    use nom_tracable::{cumulative_histogram, histogram};

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
        let e = &p.expression;
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
        let input = "(10;)";
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
            p.expression,
            Expression::Call {
                name: "print".to_string(),
                parameters: vec![
                    Expression::StringLit("Hello, World!\\n".to_string()),
                    Expression::StringLit("Testing".to_string()),
                    Expression::IntLit(1),
                    Expression::IntLit(2),
                    Expression::Nil
                ]
            },
        );
    }

    #[test]
    fn product() {
        let input = "5 * 4";
        let mut p = Program::new(input).unwrap();
        assert_eq!(
            p.expression,
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
            p.expression,
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
            p.expression,
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
            p.expression,
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
            p.expression,
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
            p.expression,
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

    #[test]
    fn lvalue() {
        let input = "a + b";
        let mut p = Program::new(input).pretty_unwrap();
        assert_eq!(
            p.expression,
            Expression::BinaryOp {
                op_type: Op::Plus,
                first: Box::new(Expression::Id("a".into())),
                second: Box::new(Expression::Id("b".into())),
            }
        );
        assert!(matches!(p.run(), Err(InterpretterError::NameError(_))));
    }

    #[test]
    fn assign() {
        let input = "a := 5";
        let mut p = Program::new(input).pretty_unwrap();
        assert_eq!(
            p.expression,
            Expression::Assign("a".into(), Box::new(Expression::IntLit(5)),)
        );
        assert!(matches!(p.run(), Err(InterpretterError::NameError(_))));
    }

    #[test]
    fn simple_declaration() {
        let input = "var a: int := 0";
        let d = Declaration::parse(Span::new_extra(input, ParserInfo::default())).finish();
        histogram();
        cumulative_histogram();
        let d = d
            .map_err(|e| FormattedError::from_nom(input, e))
            .pretty_unwrap()
            .1;
        assert_eq!(
            d,
            Declaration::Variable {
                id: "a".into(),
                type_id: Some("int".into()),
                expr: Expression::IntLit(0),
            }
        );
    }

    #[test]
    fn let_assign() {
        let input = "let
                var a: int := 0
            in
                a := 5;
                a
            end";
        let mut p = Program::new(input).pretty_unwrap();
        assert_eq!(
            p.expression,
            Expression::Let {
                decls: vec![Declaration::Variable {
                    id: "a".into(),
                    type_id: Some("int".into()),
                    expr: Expression::IntLit(0),
                }],
                exprs: Box::new(Expression::Block(vec![
                    Expression::Assign("a".into(), Box::new(Expression::IntLit(5)),),
                    Expression::Id("a".into())
                ])),
            },
        );
        assert!(matches!(p.run(), Ok(Expression::IntLit(5))));
    }

    // TODO test nested lets

    #[test]
    fn var_type() {
        let input = " : int ";
        let ty = Declaration::var_type(Span::new_extra(input, ParserInfo::default()))
            .finish()
            .map_err(|e| FormattedError::from_nom(input, e))
            .pretty_unwrap()
            .1;
        histogram();
        cumulative_histogram();
        assert_eq!(ty, Some("int".into()))
    }
}
