#![allow(non_snake_case)]
//! TriviumDB 单元测试统一入口
//!
//! 所有外部单元测试按模块组织在此目录下。
//! 每个子模块对应一个源码模块的点到点测试。

mod filter;
mod vector;
mod index;
mod core;
mod wal;
mod cognitive;
mod memtable;
mod traversal;
mod tql_ast;
mod database;
