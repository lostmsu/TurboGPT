use std::fmt;

/// One message-carrying error for the whole program, like the C++ runtime_error.
#[derive(Debug)]
pub struct Error(pub String);

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Error(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error(error.to_string())
    }
}

impl From<cuda_core::DriverError> for Error {
    fn from(error: cuda_core::DriverError) -> Self {
        Error(format!("CUDA driver: {error:?}"))
    }
}

pub fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::new(message))
}
