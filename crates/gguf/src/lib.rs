//! GGUF file parsing, mmap and tensor views.
//!
//! `unsafe` is denied here and allowed only in the mmap module.
#![deny(unsafe_code)]
