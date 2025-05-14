import NoemaBridge.Basic

namespace NoemaBridge.Category

structure SmallCategory where
  Obj : Type
  Hom : Obj → Obj → Type
  id : (X : Obj) → Hom X X
  comp : {X Y Z : Obj} → Hom X Y → Hom Y Z → Hom X Z
  id_comp : ∀ {X Y : Obj} (f : Hom X Y), comp (id X) f = f
  comp_id : ∀ {X Y : Obj} (f : Hom X Y), comp f (id Y) = f
  comp_assoc : ∀ {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
    comp (comp f g) h = comp f (comp g h)

structure CatFunctor (C D : SmallCategory) where
  obj_map : C.Obj → D.Obj
  hom_map : {X Y : C.Obj} → C.Hom X Y → D.Hom (obj_map X) (obj_map Y)
  map_id : ∀ (X : C.Obj), hom_map (C.id X) = D.id (obj_map X)
  map_comp : ∀ {X Y Z : C.Obj} (f : C.Hom X Y) (g : C.Hom Y Z),
    hom_map (C.comp f g) = D.comp (hom_map f) (hom_map g)

def idCatFunctor (C : SmallCategory) : CatFunctor C C where
  obj_map := fun X => X
  hom_map := fun f => f
  map_id := by intro X; rfl
  map_comp := by intro X Y Z f g; rfl

def compCatFunctor {C D E : SmallCategory} (G : CatFunctor D E) (F : CatFunctor C D) : CatFunctor C E where
  obj_map := fun X => G.obj_map (F.obj_map X)
  hom_map := fun f => G.hom_map (F.hom_map f)
  map_id := by
    intro X
    dsimp
    rw [F.map_id, G.map_id]
  map_comp := by
    intro X Y Z f g
    dsimp
    rw [F.map_comp, G.map_comp]

structure NatTrans {C D : SmallCategory} (F G : CatFunctor C D) where
  app : (X : C.Obj) → D.Hom (F.obj_map X) (G.obj_map X)
  naturality : ∀ {X Y : C.Obj} (f : C.Hom X Y),
    D.comp (F.hom_map f) (app Y) = D.comp (app X) (G.hom_map f)

def idNatTrans {C D : SmallCategory} (F : CatFunctor C D) : NatTrans F F where
  app := fun X => D.id (F.obj_map X)
  naturality := by
    intro X Y f
    rw [D.comp_id, D.id_comp]

def compNatTrans {C D : SmallCategory} {F G H : CatFunctor C D}
    (η : NatTrans G H) (θ : NatTrans F G) : NatTrans F H where
  app := fun X => D.comp (θ.app X) (η.app X)
  naturality := by
    intro X Y f
    dsimp
    rw [← D.comp_assoc, θ.naturality, D.comp_assoc, η.naturality, ← D.comp_assoc]

def Opposite (C : SmallCategory) : SmallCategory where
  Obj := C.Obj
  Hom := fun X Y => C.Hom Y X
  id := C.id
  comp := fun f g => C.comp g f
  id_comp := by intro X Y f; exact C.comp_id f
  comp_id := by intro X Y f; exact C.id_comp f
  comp_assoc := by intro W X Y Z f g h; exact (C.comp_assoc h g f).symm

theorem op_op_id (C : SmallCategory) : (Opposite (Opposite C)).Obj = C.Obj := by
  rfl

end NoemaBridge.Category
