//! TQL 词法分析器
//!
//! 扩展自现有 Cypher Lexer，新增支持：
//! - 关键字: FIND, SEARCH, VECTOR, TOP, EXPAND, MATCHES, ORDER, BY, ASC, DESC, OFFSET, NOT
//! - Phase 2 新增: DISTINCT, AS, OPTIONAL, COUNT, SUM, AVG, MIN, MAX, COLLECT
//! - 操作符: `$eq`, `$gt` 等 Mongo 操作符（作为特殊标识符）
//! - 符号: `|`（多标签 OR）, `*`（可变长）, `..`（范围）

#[derive(Debug, Clone, PartialEq)]
pub enum TqlToken {
    // ── 关键字 (继承) ──
    Match,
    Where,
    Return,
    Limit,
    And,
    Or,

    // ── 关键字 (TQL 新增) ──
    Find,
    Search,
    Vector,
    Top,
    Expand,
    Matches,
    Order,
    By,
    Asc,
    Desc,
    Offset,
    Not,

    // ── 关键字 (Phase 2 新增) ──
    Distinct,
    As,
    Optional,
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Collect,
    Explain,
    Create,
    Set,
    Delete,
    Detach,

    // ── 标识符 & 字面量 ──
    Ident(String),
    /// $eq, $gt 等 Mongo 操作符（含 $ 前缀）
    DollarOp(String),
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),
    BoolLit(bool),
    Null,

    // ── 符号 ──
    LParen,    // (
    RParen,    // )
    LBracket,  // [
    RBracket,  // ]
    LBrace,    // {
    RBrace,    // }
    Colon,     // :
    Dot,       // .
    DotDot,    // ..
    Comma,     // ,
    Arrow,     // ->
    LeftArrow, // <-
    Dash,      // -
    Pipe,      // |
    Star,      // *

    // ── 比较运算符 ──
    Eq,  // ==
    Ne,  // !=
    Gte, // >=
    Lte, // <=
    Gt,  // >
    Lt,  // <

    Eof,
}

pub struct TqlLexer {
    chars: Vec<char>,
    pos: usize,
}

impl TqlLexer {
    pub fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_ahead(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied();
        self.pos += 1;
        ch
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// 跳过单行注释 (-- 开头)
    fn skip_comment(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == '\n' {
                break;
            }
            self.advance();
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<TqlToken>, String> {
        let mut tokens = Vec::new();

        loop {
            self.skip_whitespace();

            match self.peek() {
                None => {
                    tokens.push(TqlToken::Eof);
                    break;
                }
                Some(ch) => {
                    let tok = match ch {
                        '(' => {
                            self.advance();
                            TqlToken::LParen
                        }
                        ')' => {
                            self.advance();
                            TqlToken::RParen
                        }
                        '[' => {
                            self.advance();
                            TqlToken::LBracket
                        }
                        ']' => {
                            self.advance();
                            TqlToken::RBracket
                        }
                        '{' => {
                            self.advance();
                            TqlToken::LBrace
                        }
                        '}' => {
                            self.advance();
                            TqlToken::RBrace
                        }
                        ':' => {
                            self.advance();
                            TqlToken::Colon
                        }
                        ',' => {
                            self.advance();
                            TqlToken::Comma
                        }
                        '|' => {
                            self.advance();
                            TqlToken::Pipe
                        }
                        '*' => {
                            self.advance();
                            TqlToken::Star
                        }

                        '.' => {
                            self.advance();
                            if self.peek() == Some('.') {
                                self.advance();
                                TqlToken::DotDot
                            } else {
                                TqlToken::Dot
                            }
                        }

                        '-' => {
                            self.advance();
                            if self.peek() == Some('>') {
                                self.advance();
                                TqlToken::Arrow
                            } else if self.peek() == Some('-') {
                                // 单行注释: --
                                self.advance();
                                self.skip_comment();
                                continue;
                            } else {
                                // 检查是否是负数: - 后面紧跟数字
                                if let Some(c) = self.peek() {
                                    if c.is_ascii_digit() {
                                        let num_tok = self.read_number()?;
                                        match num_tok {
                                            TqlToken::IntLit(n) => TqlToken::IntLit(-n),
                                            TqlToken::FloatLit(f) => TqlToken::FloatLit(-f),
                                            _ => TqlToken::Dash,
                                        }
                                    } else {
                                        TqlToken::Dash
                                    }
                                } else {
                                    TqlToken::Dash
                                }
                            }
                        }

                        '=' => {
                            self.advance();
                            if self.peek() == Some('=') {
                                self.advance();
                                TqlToken::Eq
                            } else {
                                return Err("Expected '==' but got '='".into());
                            }
                        }

                        '!' => {
                            self.advance();
                            if self.peek() == Some('=') {
                                self.advance();
                                TqlToken::Ne
                            } else {
                                return Err("Expected '!=' but got '!'".into());
                            }
                        }

                        '>' => {
                            self.advance();
                            if self.peek() == Some('=') {
                                self.advance();
                                TqlToken::Gte
                            } else {
                                TqlToken::Gt
                            }
                        }

                        '<' => {
                            self.advance();
                            if self.peek() == Some('=') {
                                self.advance();
                                TqlToken::Lte
                            } else if self.peek() == Some('-') {
                                self.advance();
                                TqlToken::LeftArrow
                            } else {
                                TqlToken::Lt
                            }
                        }

                        '"' | '\'' => {
                            let quote = ch;
                            self.advance();
                            let mut s = String::new();
                            loop {
                                match self.advance() {
                                    Some('\\') => {
                                        // 转义字符支持
                                        match self.advance() {
                                            Some('n') => s.push('\n'),
                                            Some('t') => s.push('\t'),
                                            Some('\\') => s.push('\\'),
                                            Some(c) if c == quote => s.push(c),
                                            Some(c) => {
                                                s.push('\\');
                                                s.push(c);
                                            }
                                            None => return Err("Unterminated string escape".into()),
                                        }
                                    }
                                    Some(c) if c == quote => break,
                                    Some(c) => s.push(c),
                                    None => return Err("Unterminated string literal".into()),
                                }
                            }
                            TqlToken::StringLit(s)
                        }

                        '$' => {
                            // Mongo 操作符: $eq, $gt, $in, etc.
                            self.advance();
                            let mut name = String::from("$");
                            while let Some(c) = self.peek() {
                                if c.is_ascii_alphanumeric() || c == '_' {
                                    name.push(c);
                                    self.advance();
                                } else {
                                    break;
                                }
                            }
                            TqlToken::DollarOp(name)
                        }

                        c if c.is_ascii_digit() => self.read_number()?,

                        c if c.is_ascii_alphabetic() || c == '_' => {
                            let mut ident = String::new();
                            while let Some(c) = self.peek() {
                                if c.is_ascii_alphanumeric() || c == '_' {
                                    ident.push(c);
                                    self.advance();
                                } else {
                                    break;
                                }
                            }
                            match ident.to_uppercase().as_str() {
                                "MATCH" => TqlToken::Match,
                                "WHERE" => TqlToken::Where,
                                "RETURN" => TqlToken::Return,
                                "LIMIT" => TqlToken::Limit,
                                "AND" => TqlToken::And,
                                "OR" => TqlToken::Or,
                                "NOT" => TqlToken::Not,
                                "FIND" => TqlToken::Find,
                                "SEARCH" => TqlToken::Search,
                                "VECTOR" => TqlToken::Vector,
                                "TOP" => TqlToken::Top,
                                "EXPAND" => TqlToken::Expand,
                                "MATCHES" => TqlToken::Matches,
                                "ORDER" => TqlToken::Order,
                                "BY" => TqlToken::By,
                                "ASC" => TqlToken::Asc,
                                "DESC" => TqlToken::Desc,
                                "OFFSET" => TqlToken::Offset,
                                "DISTINCT" => TqlToken::Distinct,
                                "AS" => TqlToken::As,
                                "OPTIONAL" => TqlToken::Optional,
                                "COUNT" => TqlToken::Count,
                                "SUM" => TqlToken::Sum,
                                "AVG" => TqlToken::Avg,
                                "MIN" => TqlToken::Min,
                                "MAX" => TqlToken::Max,
                                "COLLECT" => TqlToken::Collect,
                                "EXPLAIN" => TqlToken::Explain,
                                "CREATE" => TqlToken::Create,
                                "SET" => TqlToken::Set,
                                "DELETE" => TqlToken::Delete,
                                "DETACH" => TqlToken::Detach,
                                "TRUE" => TqlToken::BoolLit(true),
                                "FALSE" => TqlToken::BoolLit(false),
                                "NULL" => TqlToken::Null,
                                _ => TqlToken::Ident(ident),
                            }
                        }

                        _ => return Err(format!("Unexpected character: '{}'", ch)),
                    };
                    tokens.push(tok);
                }
            }
        }

        Ok(tokens)
    }

    /// 读取数字（整数或浮点数）
    fn read_number(&mut self) -> Result<TqlToken, String> {
        let mut num_str = String::new();
        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                num_str.push(c);
                self.advance();
            } else if c == '.'
                && !is_float
                && self.peek_ahead(1).is_some_and(|c| c.is_ascii_digit())
            {
                // 只有 "数字.数字" 才是浮点数，"数字.." 是整数 + DotDot
                is_float = true;
                num_str.push(c);
                self.advance();
            } else {
                break;
            }
        }
        if is_float {
            Ok(TqlToken::FloatLit(
                num_str.parse().map_err(|e| format!("Bad float: {}", e))?,
            ))
        } else {
            Ok(TqlToken::IntLit(
                num_str.parse().map_err(|e| format!("Bad int: {}", e))?,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_query() {
        let mut lexer = TqlLexer::new(r#"FIND {type: "event", heat: {$gte: 0.7}} RETURN *"#);
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0], TqlToken::Find);
        assert_eq!(tokens[1], TqlToken::LBrace);
        // type: "event"
        assert_eq!(tokens[2], TqlToken::Ident("type".into()));
        assert_eq!(tokens[3], TqlToken::Colon);
        assert_eq!(tokens[4], TqlToken::StringLit("event".into()));
        // $gte: 0.7
        assert!(tokens.contains(&TqlToken::DollarOp("$gte".into())));
        assert!(tokens.contains(&TqlToken::Star));
    }

    #[test]
    fn test_match_variable_length() {
        let mut lexer = TqlLexer::new("MATCH (a)-[:knows*1..3]->(b) RETURN b");
        let tokens = lexer.tokenize().unwrap();
        assert!(tokens.contains(&TqlToken::Star));
        assert!(tokens.contains(&TqlToken::DotDot));
    }

    #[test]
    fn test_pipe_multi_label() {
        let mut lexer = TqlLexer::new("MATCH (a)-[:knows|works_at]->(b) RETURN b");
        let tokens = lexer.tokenize().unwrap();
        assert!(tokens.contains(&TqlToken::Pipe));
    }

    #[test]
    fn test_search_entry() {
        let mut lexer = TqlLexer::new("SEARCH VECTOR [0.1, -0.2, 0.3] TOP 10 RETURN *");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0], TqlToken::Search);
        assert_eq!(tokens[1], TqlToken::Vector);
        assert_eq!(tokens[2], TqlToken::LBracket);
    }

    #[test]
    fn test_comment_skip() {
        let mut lexer = TqlLexer::new("FIND {type: \"event\"} -- this is a comment\nRETURN *");
        let tokens = lexer.tokenize().unwrap();
        assert!(tokens.contains(&TqlToken::Return));
    }

    #[test]
    fn test_order_by() {
        let mut lexer = TqlLexer::new("ORDER BY a.score DESC LIMIT 10 OFFSET 20");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0], TqlToken::Order);
        assert_eq!(tokens[1], TqlToken::By);
        assert!(tokens.contains(&TqlToken::Desc));
        assert!(tokens.contains(&TqlToken::Offset));
    }

    #[test]
    fn test_negative_number() {
        let mut lexer = TqlLexer::new("[-0.5, -3]");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[1], TqlToken::FloatLit(-0.5));
        assert_eq!(tokens[3], TqlToken::IntLit(-3));
    }

    #[test]
    fn test_dot_dot_not_float() {
        // "1..3" should be IntLit(1), DotDot, IntLit(3) — NOT FloatLit(1.0) + error
        let mut lexer = TqlLexer::new("1..3");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0], TqlToken::IntLit(1));
        assert_eq!(tokens[1], TqlToken::DotDot);
        assert_eq!(tokens[2], TqlToken::IntLit(3));
    }
}
