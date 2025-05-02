import NoemaBridge.Basic

namespace NoemaBridge.Logic

theorem id_prop (P : Prop) (h : P) : P := by
  exact h

theorem imp_trans (P Q R : Prop) (hpq : P → Q) (hqr : Q → R) (hp : P) : R := by
  exact hqr (hpq hp)

theorem and_comm_iff (P Q : Prop) : (P ∧ Q) ↔ (Q ∧ P) := by
  constructor
  · intro h; exact ⟨h.2, h.1⟩
  · intro h; exact ⟨h.2, h.1⟩

theorem or_comm_iff (P Q : Prop) : (P ∨ Q) ↔ (Q ∨ P) := by
  constructor
  · intro h; cases h with | inl hp => exact Or.inr hp | inr hq => exact Or.inl hq
  · intro h; cases h with | inl hq => exact Or.inr hq | inr hp => exact Or.inl hp

theorem and_assoc_iff (P Q R : Prop) : ((P ∧ Q) ∧ R) ↔ (P ∧ (Q ∧ R)) := by
  constructor
  · intro h; exact ⟨h.1.1, ⟨h.1.2, h.2⟩⟩
  · intro h; exact ⟨⟨h.1, h.2.1⟩, h.2.2⟩

theorem de_morgan_not_or (P Q : Prop) : ¬(P ∨ Q) ↔ (¬P ∧ ¬Q) := by
  constructor
  · intro h; exact ⟨fun hp => h (Or.inl hp), fun hq => h (Or.inr hq)⟩
  · intro h hpq; cases hpq with | inl hp => exact h.1 hp | inr hq => exact h.2 hq

theorem contraposition (P Q : Prop) (hpq : P → Q) : ¬Q → ¬P := by
  intro hnq hp
  exact hnq (hpq hp)

theorem disjunctive_syllogism (P Q : Prop) (h : P ∨ Q) (hnp : ¬P) : Q := by
  cases h with
  | inl hp => exact False.elim (hnp hp)
  | inr hq => exact hq

theorem absorption_and_or (P Q : Prop) : (P ∧ (P ∨ Q)) ↔ P := by
  constructor
  · intro h; exact h.1
  · intro hp; exact ⟨hp, Or.inl hp⟩

theorem absorption_or_and (P Q : Prop) : (P ∨ (P ∧ Q)) ↔ P := by
  constructor
  · intro h; cases h with | inl hp => exact hp | inr hpq => exact hpq.1
  · intro hp; exact Or.inl hp

theorem and_distrib_or (P Q R : Prop) : (P ∧ (Q ∨ R)) ↔ ((P ∧ Q) ∨ (P ∧ R)) := by
  constructor
  · intro h
    cases h.2 with
    | inl hq => exact Or.inl ⟨h.1, hq⟩
    | inr hr => exact Or.inr ⟨h.1, hr⟩
  · intro h
    cases h with
    | inl hpq => exact ⟨hpq.1, Or.inl hpq.2⟩
    | inr hpr => exact ⟨hpr.1, Or.inr hpr.2⟩

theorem double_negation_intro (P : Prop) (hp : P) : ¬¬P := by
  intro hnp
  exact hnp hp

end NoemaBridge.Logic
