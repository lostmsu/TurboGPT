use std::time::Instant;

pub fn seconds(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

#[inline(always)]
pub fn hash_u64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// C++ iostream formatting of a floating value at `precision` significant digits (%g).
pub fn fmt_g(value: f64, precision: usize) -> String {
    if !value.is_finite() {
        return if value.is_nan() {
            "nan".into()
        } else if value > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let p = precision.max(1);
    let scientific = format!("{:.*e}", p - 1, value);
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if exponent < -4 || exponent >= p as i32 {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", strip(mantissa), sign, exponent.abs())
    } else {
        let decimals = (p as i32 - 1 - exponent).max(0) as usize;
        strip(&format!("{:.*}", decimals, value))
    }
}

#[cfg(test)]
mod tests {
    use super::fmt_g;

    #[test]
    fn formats_like_iostream() {
        assert_eq!(fmt_g(0.3, 9), "0.3");
        assert_eq!(fmt_g(4e-6, 9), "4e-06");
        assert_eq!(fmt_g(0.0006000000285, 10), "0.0006000000285");
        assert_eq!(fmt_g(129726622.2, 10), "129726622.2");
        assert_eq!(fmt_g(1500007680.0, 9), "1.50000768e+09");
        assert_eq!(fmt_g(2.52600131e-07, 9), "2.52600131e-07");
        assert_eq!(fmt_g(1.0, 6), "1");
    }
}
