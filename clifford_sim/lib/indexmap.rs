//! Non-hash-based mappings from integers to arbitrary elements.
//!
//! An [`IndexMap`] sits conceptually between a `Vec` and a `HashMap`: Like a
//! `Vec`, elements in this collection are uniquely named via assignment to
//! natural number indices; like a `HashMap`, these indices need not be adjacent
//! elements in the set of natural numbers.
//!
//! Internally, each `IndexMap<T>` uses a `VecDeque<Option<T>>` to allow for
//! efficient iteration and insertion of elements at either end of the range of
//! pre-existing indices.

use std::{ collections::VecDeque, fmt };
use itertools::Itertools;

#[derive(Clone, Debug)]
pub struct IndexMap<T> {
    pub(crate) offs: usize,
    pub(crate) data: VecDeque<Option<T>>,
    pub(crate) count: usize,
}

impl<T> FromIterator<(usize, T)> for IndexMap<T> {
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = (usize, T)>
    {
        let mut data: VecDeque<Option<T>> = VecDeque::new();
        let mut k_min = usize::MAX;
        let mut k_max = usize::MIN;
        let mut count: usize = 0;
        for (k, x) in iter.into_iter() {
            if data.is_empty() {
                data.push_back(Some(x));
                k_min = k;
                k_max = k;
                count += 1;
            } else if (k_min..=k_max).contains(&k) {
                if data[k - k_min].is_none() { count += 1; }
                data[k - k_min] = Some(x);
            } else if k < k_min {
                data.resize_with(data.len() + k_min - k, || None);
                data.rotate_right(k_min - k);
                data[0] = Some(x);
                k_min = k;
                count += 1;
            } else if k > k_max {
                data.resize_with(data.len() + k - k_max, || None);
                let last = data.back_mut().unwrap();
                *last = Some(x);
                k_max = k;
                count += 1;
            } else {
                unreachable!()
            }
        }
        if data.is_empty() { k_min = 0; }
        Self { offs: k_min, data, count }
    }
}

impl<T> IntoIterator for IndexMap<T> {
    type Item = (usize, T);
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            offs: self.offs,
            len: self.count,
            iter: self.data.into_iter().enumerate(),
        }
    }
}

impl<T> Default for IndexMap<T> {
    fn default() -> Self { Self::new() }
}

impl<T> IndexMap<T> {
    /// Create a new, empty `IndexMap`.
    pub fn new() -> Self {
        Self { offs: 0, data: VecDeque::new(), count: 0 }
    }

    /// Create a new, empty `IndexMap` with pre-allocated, contiguous indices
    /// over the range `a..=b`.
    pub fn with_capacity(a: usize, b: usize) -> Self {
        let k_min = a.min(b);
        let k_max = a.max(b);
        let mut data: VecDeque<Option<T>> =
            VecDeque::with_capacity(k_max - k_min + 1);
        data.resize_with(k_max - k_min + 1, || None);
        Self { offs: k_min, data, count: 0 }
    }

    /// Return the minimum inhabited index.
    ///
    /// Returns `None` if no indices are inhabited.
    pub fn idx_min(&self) -> Option<usize> {
        self.data.iter().enumerate()
            .find_map(|(id, mb_x)| mb_x.as_ref().map(|_| id + self.offs))
    }

    /// Return the maximum inhabited index.
    pub fn idx_max(&self) -> Option<usize> {
        self.data.iter().enumerate()
            .rfind(|(_, mb_x)| mb_x.is_some())
            .map(|(id, _)| id + self.offs)
    }

    /// Return the maximal `a` and minimal `b` such that all inhabited indices
    /// fall in the range(`a..=b`.
    ///
    /// Returns `None` if no indices are inhabited.
    pub fn idx_bounds(&self) -> Option<(usize, usize)> {
        let k_min = self.idx_min()?;
        let k_max = self.idx_max()?;
        Some((k_min, k_max))
    }

    /// Return the maximal `a` and minimal `b` such that all inhabited indices
    /// are elements in the output range.
    ///
    /// Returns `None` if no indices are inhabited.
    pub fn idx_range(&self) -> Option<std::ops::RangeInclusive<usize>> {
        let k_min = self.idx_min()?;
        let k_max = self.idx_max()?;
        Some(k_min..=k_max)
    }

    /// Shift all indices to the right (larger values) by a uniform distance.
    ///
    /// *Panics if the size of the shift would cause any indices to overflow
    /// `usize`*.
    pub fn shift_indices_r(&mut self, sh: usize) {
        if let Some(k_max) = self.idx_max() {
            k_max.checked_add(sh)
                .expect("IndexMap::shift_indices_r: shift would cause index overflow");
        }
        self.offs += sh;
    }

    /// Shift all indices to the left (smaller values) by a uniform distance.
    ///
    /// *Panics if the size of the shift would cause any indices to underflow
    /// `usize`*.
    pub fn shift_indices_l(&mut self, sh: usize) {
        if let Some(k_min) = self.idx_min() {
            k_min.checked_sub(sh)
                .expect("IndexMap::shift_indices_l: shift would cause index underflow");
        }
        if self.offs < sh {
            (0..sh - self.offs).for_each(|_| { self.data.pop_front(); });
            self.offs = 0;
        } else {
            self.offs -= sh;
        }
    }

    /// Get a reference to the element at index `k`, if it exists.
    pub fn get(&self, k: usize) -> Option<&T> {
        if let Some(range) = self.idx_range() {
            if range.contains(&k) {
                self.data[k - self.offs].as_ref()
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get a mutable reference to the element at index `k`, if it exists.
    pub fn get_mut(&mut self, k: usize) -> Option<&mut T> {
        if let Some(range) = self.idx_range() {
            if range.contains(&k) {
                self.data[k - self.offs].as_mut()
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Insert an element at index `k`.
    ///
    /// Existing elements are replaced and returned, if they exist.
    pub fn insert(&mut self, k: usize, elem: T) -> Option<T> {
        if self.data.is_empty() {
            self.offs = k;
            self.data.push_back(Some(elem));
            self.count += 1;
            None
        } else if k < self.offs {
            self.data.resize_with(self.data.len() + self.offs - k, || None);
            self.data.rotate_right(self.offs - k);
            self.data[0] = Some(elem);
            self.offs = k;
            self.count += 1;
            None
        } else if k > self.offs + (self.data.len() - 1) {
            self.data.resize_with(k - self.offs + 1, || None);
            let last = self.data.back_mut().unwrap();
            *last = Some(elem);
            self.count += 1;
            None
        } else {
            let ret = self.data[k - self.offs].replace(elem);
            if ret.is_none() { self.count += 1; }
            ret
        }
    }

    /// Remove and return the element at index `k`, if it exists.
    pub fn remove(&mut self, k: usize) -> Option<T> {
        if self.data.is_empty() {
            None
        } else if (self.offs..=self.offs + (self.data.len() - 1)).contains(&k) {
            let ret = self.data[k - self.offs].take();
            if ret.is_some() { self.count -= 1; }
            ret
        } else {
            None
        }
    }

    /// Swap the elements at indices `a` and `b`.
    ///
    /// If only one of either exist, then the existing element is moved to the
    /// new index. If neither exist, no operation is performed.
    pub fn swap(&mut self, a: usize, b: usize) {
        if a == b
            || !(self.contains_key(a) || self.contains_key(b))
            || self.data.is_empty()
        {
            return;
        }
        let range = self.offs..=self.offs + (self.data.len() - 1);
        let k_min = a.min(b);
        let k_max = a.max(b);
        if !range.contains(&k_min) {
            self.data.resize_with(self.data.len() + self.offs - k_min, || None);
            self.data.rotate_right(self.offs - k_min);
        }
        if !range.contains(&k_max) {
            self.data.resize_with(k_max + 1 - self.offs, || None);
        }
        self.data.swap(a - self.offs, b - self.offs);
    }

    /// Return the number of inhabited indices.
    pub fn len(&self) -> usize { self.count }

    /// Return `true` if no indices are occupied.
    pub fn is_empty(&self) -> bool { self.count == 0 }

    /// Return an iterator over all occupied indices.
    ///
    /// The iterator item type is `(usize, &T)`.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            offs: self.offs,
            len: self.count,
            iter: self.data.iter().enumerate(),
        }
    }

    /// Return an iterator over mutable references to all occupied indices.
    ///
    /// The iterator item type is `(usize, &mut T)`.
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            offs: self.offs,
            len: self.count,
            iter: self.data.iter_mut().enumerate(),
        }
    }

    /// Return `true` the index `k` is inhabited.
    pub fn contains_key(&self, k: usize) -> bool {
        if self.data.is_empty() {
            false
        } else if (self.offs..=self.offs + (self.data.len() - 1)).contains(&k) {
            self.data[k - self.offs].is_some()
        } else {
            false
        }
    }
}

impl<T> fmt::Display for IndexMap<T>
where T: fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        let n = self.len();
        for (k, (id, x)) in self.iter().enumerate() {
            write!(f, "{}: ", id)?;
            x.fmt(f)?;
            if k < n - 1 { write!(f, ", ")?; }
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<T> PartialEq for IndexMap<T>
where T: PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        if self.count != other.count {
            false
        } else {
            self.iter().zip(other.iter())
                .all(|((id_l, x_l), (id_r, x_r))| id_l == id_r && x_l == x_r)
        }
    }
}

impl<T> Eq for IndexMap<T>
where T: Eq
{ }

impl<T> PartialOrd for IndexMap<T>
where T: PartialOrd
{
    fn partial_cmp(&self, rhs: &Self) -> Option<std::cmp::Ordering> {
        use itertools::EitherOrBoth::*;
        for mb_pair in self.iter().zip_longest(rhs.iter()) {
            match mb_pair {
                Right(_) => { return Some(std::cmp::Ordering::Less); },
                Left(_) => { return Some(std::cmp::Ordering::Greater); },
                Both((id_l, x_l), (id_r, x_r)) => {
                    match id_l.cmp(&id_r) {
                        std::cmp::Ordering::Equal => {
                            match x_l.partial_cmp(x_r)? {
                                std::cmp::Ordering::Equal => { continue; },
                                ord => { return Some(ord); },
                            }
                        },
                        idx_ord => { return Some(idx_ord); },
                    }
                }
            }
        }
        Some(std::cmp::Ordering::Equal)
    }
}

impl<T> Ord for IndexMap<T>
where T: Ord
{
    fn cmp(&self, rhs: &Self) -> std::cmp::Ordering {
        use itertools::EitherOrBoth::*;
        for mb_pair in self.iter().zip_longest(rhs.iter()) {
            match mb_pair {
                Right(_) => { return std::cmp::Ordering::Less; },
                Left(_) => { return std::cmp::Ordering::Greater; },
                Both((id_l, x_l), (id_r, x_r)) => {
                    if id_l < id_r {
                        return std::cmp::Ordering::Less;
                    } else if id_l > id_r {
                        return std::cmp::Ordering::Greater;
                    } else if x_l < x_r {
                        return std::cmp::Ordering::Less;
                    } else if x_l > x_r {
                        return std::cmp::Ordering::Greater;
                    } else {
                        continue;
                    }
                },
            }
        }
        std::cmp::Ordering::Equal
    }
}

/// Iterator over the inhabited indices of an [`IndexMap`].
///
/// The iterator item type is `(usize, &T)`.
#[derive(Clone, Debug)]
pub struct Iter<'a, T> {
    offs: usize,
    len: usize,
    iter: std::iter::Enumerate<std::collections::vec_deque::Iter<'a, Option<T>>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|(id, mb_x)| {
            mb_x.as_ref()
                .map(|x| {
                    self.len = self.len.saturating_sub(1);
                    (self.offs + id, x)
                })
        })
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.rfind(|(_, mb_x)| mb_x.is_some())
            .map(|(id, mb_x)| {
                self.len = self.len.saturating_sub(1);
                (self.offs + id, mb_x.as_ref().unwrap())
            })
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {
    fn len(&self) -> usize { self.len }
}

impl<'a, T> std::iter::FusedIterator for Iter<'a, T> { }

/// Iterator over mutable references to the inhabited indices of an
/// [`IndexMap`].
///
/// The iterator item type is `(usize, &mut T)`.
#[derive(Debug)]
pub struct IterMut<'a, T> {
    offs: usize,
    len: usize,
    iter: std::iter::Enumerate<std::collections::vec_deque::IterMut<'a, Option<T>>>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (usize, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|(id, mb_x)| {
            mb_x.as_mut()
                .map(|x| {
                    self.len = self.len.saturating_sub(1);
                    (self.offs + id, x)
                })
        })
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.rfind(|(_, mb_x)| mb_x.is_some())
            .map(|(id, mb_x)| {
                self.len = self.len.saturating_sub(1);
                (self.offs + id, mb_x.as_mut().unwrap())
            })
    }
}

impl<'a, T> ExactSizeIterator for IterMut<'a, T> {
    fn len(&self) -> usize { self.len }
}

impl<'a, T> std::iter::FusedIterator for IterMut<'a, T> { }

/// Iterator over all inhabited indices of an [`IndexMap`].
///
/// The iterator item type is `(usize, T)`.
#[derive(Clone, Debug)]
pub struct IntoIter<T> {
    offs: usize,
    len: usize,
    iter: std::iter::Enumerate<std::collections::vec_deque::IntoIter<Option<T>>>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|(id, mb_x)| {
            mb_x.map(|x| {
                self.len = self.len.saturating_sub(1);
                (self.offs + id, x)
            })
        })
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.rfind(|(_, mb_x)| mb_x.is_some())
            .map(|(id, mb_x)| {
                self.len = self.len.saturating_sub(1);
                (self.offs + id, mb_x.unwrap())
            })
    }
}

impl<T> std::iter::FusedIterator for IntoIter<T> { }

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn from_iter() {
        let elems: IndexMap<char> =
            [
                (2, 'a'),
                (5, 'c'),
                (3, 'b'),
                (8, 'd'),
            ]
            .into_iter()
            .collect();
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('b'),
                None,
                Some('c'),
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.offs, 2);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 4);

        let elems: IndexMap<char> =
            [
                (1, 'a'),
                (1, 'c'),
                (1, 'd'),
                (1, 'b'),
            ]
            .into_iter()
            .collect();
        let data_expected: VecDeque<Option<char>> =
            vec![Some('b')].into();
        assert_eq!(elems.offs, 1);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 1);
    }

    #[test]
    fn with_capacity() {
        let elems: IndexMap<char> = IndexMap::with_capacity(5, 7);
        let data_expected: VecDeque<Option<char>> =
            vec![None, None, None].into();
        assert_eq!(elems.offs, 5);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 0);
    }

    #[test]
    fn insert() {
        let mut elems: IndexMap<char> = IndexMap::new();
        elems.insert(2, 'a');
        elems.insert(5, 'c');
        elems.insert(3, 'b');
        elems.insert(8, 'd');
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('b'),
                None,
                Some('c'),
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.offs, 2);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 4);
    }

    fn build_simple() -> IndexMap<char> {
        let mut elems: IndexMap<char> = IndexMap::new();
        elems.insert(2, 'a');
        elems.insert(1, 'a');
        elems.insert(5, 'c');
        elems.insert(3, 'b');
        elems.insert(8, 'd');
        elems
    }

    #[test]
    fn remove() {
        let mut elems = build_simple();
        assert_eq!(elems.remove(4), None);
        assert_eq!(elems.remove(9), None);
        assert_eq!(elems.remove(0), None);
        assert_eq!(elems.remove(3), Some('b'));
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('a'),
                None,
                None,
                Some('c'),
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.offs, 1);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 4);
    }

    #[test]
    fn idx_range() {
        let mut elems = build_simple();
        assert_eq!(elems.idx_range(), Some(1..=8));
        elems.remove(2);
        assert_eq!(elems.idx_range(), Some(1..=8));
        elems.remove(1);
        assert_eq!(elems.idx_range(), Some(3..=8));
        elems.remove(8);
        assert_eq!(elems.idx_range(), Some(3..=5));
        elems.remove(3);
        elems.remove(5);
        assert_eq!(elems.idx_range(), None);
    }

    #[test]
    fn shift() {
        let mut elems = build_simple();
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('a'),
                Some('b'),
                None,
                Some('c'),
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.offs, 1);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 5);

        elems.shift_indices_l(1);
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('a'),
                Some('b'),
                None,
                Some('c'),
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.offs, 0);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 5);

        elems.shift_indices_r(5);
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('a'),
                Some('b'),
                None,
                Some('c'),
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.offs, 5);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 5);

        elems.insert(1, 'b');
        elems.remove(1);
        let data_expected: VecDeque<Option<char>> =
            vec![
                None,
                None,
                None,
                None,
                Some('a'),
                Some('a'),
                Some('b'),
                None,
                Some('c'),
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.offs, 1);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 5);

        elems.shift_indices_l(3);
        let data_expected: VecDeque<Option<char>> =
            vec![
                None,
                None,
                Some('a'),
                Some('a'),
                Some('b'),
                None,
                Some('c'),
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.offs, 0);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 5);

        elems.shift_indices_l(2);
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('a'),
                Some('b'),
                None,
                Some('c'),
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.offs, 0);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 5);

        elems.shift_indices_r(usize::MAX - 8);
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('a'),
                Some('b'),
                None,
                Some('c'),
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.offs, usize::MAX - 8);
        assert_eq!(elems.data, data_expected);
        assert_eq!(elems.count, 5);

        assert!(elems.get(usize::MAX - 1).is_some());
    }

    #[test]
    #[should_panic]
    fn bad_shift_l() {
        let mut elems = build_simple();
        elems.shift_indices_l(2);
    }

    #[test]
    #[should_panic]
    fn bad_shift_r() {
        let mut elems = build_simple();
        elems.shift_indices_r(usize::MAX - 6);
    }

    #[test]
    fn get() {
        let elems = build_simple();
        assert_eq!(elems.get(0), None);
        assert_eq!(elems.get(1), Some(&'a'));
        assert_eq!(elems.get(4), None);
        assert_eq!(elems.get(10), None);
    }

    #[test]
    fn contains_key() {
        let elems = build_simple();
        assert_eq!(elems.contains_key(0), false);
        assert_eq!(elems.contains_key(1), true);
        assert_eq!(elems.contains_key(4), false);
        assert_eq!(elems.contains_key(10), false);
    }

    #[test]
    fn eq() {
        let elems0 = build_simple();
        let mut elems1 = build_simple();
        elems1.insert(10, 'c');
        elems1.remove(10);
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('a'),
                Some('b'),
                None,
                Some('c'),
                None,
                None,
                Some('d'),
                None,
                None,
            ]
            .into();
        assert_eq!(elems1.offs, 1);
        assert_eq!(elems1.data, data_expected);
        assert_eq!(elems1.count, 5);
        assert_eq!(elems0, elems1);
    }

    #[test]
    fn swap() {
        let mut elems = build_simple();
        elems.swap(8, 10);
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('a'),
                Some('b'),
                None,
                Some('c'),
                None,
                None,
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.data, data_expected);
        elems.swap(2, 5);
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('c'),
                Some('b'),
                None,
                Some('a'),
                None,
                None,
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.data, data_expected);
        elems.swap(0, 20);
        let data_expected: VecDeque<Option<char>> =
            vec![
                Some('a'),
                Some('c'),
                Some('b'),
                None,
                Some('a'),
                None,
                None,
                None,
                None,
                Some('d'),
            ]
            .into();
        assert_eq!(elems.data, data_expected);
    }

    #[test]
    fn len_is_empty() {
        let mut elems = build_simple();
        assert_eq!(elems.len(), 5);
        assert_eq!(elems.is_empty(), false);
        elems.remove(1);
        assert_eq!(elems.len(), 4);
        assert_eq!(elems.is_empty(), false);
        elems.remove(0);
        assert_eq!(elems.len(), 4);
        assert_eq!(elems.is_empty(), false);
        elems.remove(2);
        assert_eq!(elems.len(), 3);
        assert_eq!(elems.is_empty(), false);
        elems.remove(3);
        assert_eq!(elems.len(), 2);
        assert_eq!(elems.is_empty(), false);
        elems.remove(4);
        assert_eq!(elems.len(), 2);
        assert_eq!(elems.is_empty(), false);
        elems.remove(5);
        elems.remove(8);
        assert_eq!(elems.len(), 0);
        assert_eq!(elems.is_empty(), true);
    }

    #[test]
    fn iter() {
        let elems = build_simple();
        let items: Vec<(usize, &char)> = elems.iter().collect();
        let items_expected: Vec<(usize, &char)> =
            vec![
                (1, &'a'),
                (2, &'a'),
                (3, &'b'),
                (5, &'c'),
                (8, &'d'),
            ];
        assert_eq!(items, items_expected);
    }

    #[test]
    fn into_iter() {
        let elems = build_simple();
        let items: Vec<(usize, char)> = elems.into_iter().collect();
        let items_expected: Vec<(usize, char)> =
            vec![
                (1, 'a'),
                (2, 'a'),
                (3, 'b'),
                (5, 'c'),
                (8, 'd'),
            ];
        assert_eq!(items, items_expected);
    }

}

