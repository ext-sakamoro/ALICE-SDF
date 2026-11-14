//! A last-in first-out stack for the tree walkers that do not recurse
//!
//! The first `N` entries live inline (no allocation for the depths ordinary scenes have); deeper
//! trees spill the rest to a `Vec`, so a walker's depth is bounded by the heap, not by the stack
//! of the thread or of a wasm module.
//!
//! Author: Moroya Sakamoto

use arrayvec::ArrayVec;

pub struct WorkStack<T, const N: usize> {
    inline: ArrayVec<T, N>,
    spill: Vec<T>,
}

impl<T, const N: usize> WorkStack<T, N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            inline: ArrayVec::new_const(),
            spill: Vec::new(),
        }
    }

    /// Push on top: inline while there is room and nothing has spilled, else on the spill
    #[inline]
    pub fn push(&mut self, value: T) {
        if self.spill.is_empty() {
            if let Err(e) = self.inline.try_push(value) {
                self.spill.push(e.element());
            }
        } else {
            self.spill.push(value);
        }
    }

    /// Pop the top: the spill holds everything pushed after the inline part filled up
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.spill.pop().or_else(|| self.inline.pop())
    }

    #[cfg(test)]
    const fn len(&self) -> usize {
        self.inline.len() + self.spill.len()
    }
}

#[cfg(test)]
mod tests {
    use super::WorkStack;

    /// Last in, first out across the inline part and the spill, for every count around the
    /// inline capacity, interleaving pushes and pops
    #[test]
    fn lifo_across_the_spill() {
        for total in [0usize, 1, 3, 4, 5, 9, 64] {
            let mut s: WorkStack<usize, 4> = WorkStack::new();
            for i in 0..total {
                s.push(i);
            }
            assert_eq!(s.len(), total);
            // pop half, push them back, then drain
            let half: Vec<usize> = (0..total / 2).filter_map(|_| s.pop()).collect();
            for &v in half.iter().rev() {
                s.push(v);
            }
            let drained: Vec<usize> = std::iter::from_fn(|| s.pop()).collect();
            assert_eq!(
                drained,
                (0..total).rev().collect::<Vec<_>>(),
                "total {total}"
            );
            assert_eq!(s.pop(), None);
        }
    }
}
