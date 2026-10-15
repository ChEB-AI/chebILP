% ===========================================================================
% core.lp - shared chemical helper predicates over the parse_molecules.py BK.
% Atom-level predicates take atom ids, molecule-level ones (has_*) take M.
% Only statements needed by a class are copied into its program (asp_tools).
% ===========================================================================

% ---------------------------------------------------------------- elements
carbon(A)     :- element(A,c).
hydrogen(A)   :- element(A,h).
nitrogen(A)   :- element(A,n).
oxygen(A)     :- element(A,o).
sulfur(A)     :- element(A,s).
phosphorus(A) :- element(A,p).
selenium(A)   :- element(A,se).
rgroup(A)     :- element(A,r).
heavy(A)      :- element(A,E), E != h.
halogen_el(f;cl;br;i;at).
halogen(A)    :- element(A,E), halogen_el(E).
chalcogen_el(o;s;se;te;po).
chalcogen(A)  :- element(A,E), chalcogen_el(E).
pnictogen_el(n;p;as;sb;bi).
pnictogen(A)  :- element(A,E), pnictogen_el(E).
metal_el(li;na;k;rb;cs;fr;be;mg;ca;sr;ba;ra;al;ga;in;tl;sn;pb;bi;po;
         sc;ti;v;cr;mn;fe;co;ni;cu;zn;y;zr;nb;mo;tc;ru;rh;pd;ag;cd;
         la;ce;pr;nd;pm;sm;eu;gd;tb;dy;ho;er;tm;yb;lu;hf;ta;w;re;os;ir;pt;au;hg;
         ac;th;pa;u;np;pu;am;cm;bk;cf;es;fm;md;no;lr).
metal(A)      :- element(A,E), metal_el(E).
alkali_el(li;na;k;rb;cs;fr).
alkali_metal(A) :- element(A,E), alkali_el(E).
% carbon or an unspecified substituent R (class-level structures with R groups)
carbon_or_r(A) :- carbon(A).
carbon_or_r(A) :- rgroup(A).
heteroatom(A) :- element(A,E), E != c, E != h.

% ---------------------------------------------------------------- bonds
nb(A,B)      :- bond(A,B,_).
single(A,B)  :- bond(A,B,single).
double(A,B)  :- bond(A,B,double).
triple(A,B)  :- bond(A,B,triple).
arombond(A,B):- bond(A,B,aromatic).
% single or aromatic bond (RDKit writes e.g. coumarin ring C-O or pyridone C-N as aromatic)
sa(A,B) :- bond(A,B,single).
sa(A,B) :- bond(A,B,aromatic).
% atom A has a non-single bond (double/triple/aromatic) -> not sp3
unsat_atom(A) :- bond(A,_,T), T != single, T != dative.
saturated_c(A) :- carbon(A), not unsat_atom(A), not aromatic(A).
% number of heavy (non-H) neighbours
heavy_degree(A,N) :- heavy(A), N = #count{ B : nb(A,B), heavy(B) }.
% number of carbon neighbours
c_neighbors(A,N) :- heavy(A), N = #count{ B : nb(A,B), carbon(B) }.
% number of hetero (non C, non H) neighbours
hetero_neighbors(A,N) :- heavy(A), N = #count{ B : nb(A,B), heteroatom(B) }.

% ---------------------------------------------------------------- oxygen groups
% carbonyl C=O (any carbon)
carbonyl(C,O) :- carbon(C), double(C,O), oxygen(O).
carbonyl_c(C) :- carbonyl(C,_).
% hydroxy group -OH (neutral O with one H and one heavy neighbour X)
hydroxy(O,X) :- oxygen(O), hcount(O,1), charge(O,0), nb(O,X), heavy(X).
hydroxy_o(O) :- hydroxy(O,_).
% oxido -O(-) singly bonded to X
oxido(O,X) :- oxygen(O), charge(O,-1), hcount(O,0), single(O,X), heavy_degree(O,1).
% OH or O(-) on X ("acidic oxygen" of oxoacids and their conjugate bases)
oh_or_ominus(O,X) :- hydroxy(O,X).
oh_or_ominus(O,X) :- oxido(O,X).

% carboxy carbon: C(=O)(-O[H/-])-X with X carbon/R or H (formic acid)
% "carbonyl carbon whose remaining substituent(s) besides heteroatoms are C/R/H"
carboxy_core(C,O1,O2) :- carbonyl(C,O1), sa(C,O2), oxygen(O2), O1 != O2,
    #count{ B : nb(C,B), B != O1, B != O2, not carbon_or_r(B) } = 0.
carboxylic_acid_group(C) :- carboxy_core(C,_,O2), hydroxy(O2,C).
carboxylate_group(C)     :- carboxy_core(C,_,O2), oxido(O2,C).
carboxy_group(C) :- carboxylic_acid_group(C).
carboxy_group(C) :- carboxylate_group(C).
% carboxylic ester: R-C(=O)-O-R' (R' carbon/R, O not in a ring -> lactones below)
carboxylic_ester(C,O2,R2) :- carboxy_core(C,_,O2), nb(O2,R2), R2 != C, carbon_or_r(R2).
lactone(C,O2) :- carboxylic_ester(C,O2,_), ring_bond(C,O2).
% carboxylic anhydride C(=O)-O-C(=O)
carboxylic_anhydride(O) :- carboxy_core(C1,_,O), carboxy_core(C2,_,O), C1 != C2.

% aldehyde: C(=O)H with the other substituent C/R (or H: formaldehyde)
aldehyde_group(C) :- carbonyl(C,O), hcount(C,H), H >= 1,
    #count{ B : nb(C,B), B != O, not carbon_or_r(B) } = 0.
% ketone: C(=O) bonded to exactly two carbons (incl. R groups)
ketone_group(C) :- carbonyl(C,O), hcount(C,0), not charged(C),
    #count{ B : nb(C,B), B != O, carbon_or_r(B) } = 2,
    #count{ B : nb(C,B), B != O } = 2.
charged(A) :- charge(A,Q), Q != 0.

% alcohol: OH on a saturated carbon
alcohol_oh(O,C) :- hydroxy(O,C), saturated_c(C).
primary_alcohol_oh(O,C)   :- alcohol_oh(O,C), c_neighbors(C,N), N <= 1, hcount(C,H), H >= 2.
secondary_alcohol_oh(O,C) :- alcohol_oh(O,C), c_neighbors(C,2), hcount(C,1).
tertiary_alcohol_oh(O,C)  :- alcohol_oh(O,C), c_neighbors(C,3).
% phenol-type OH: OH on an aromatic carbon of a carbocyclic aromatic ring
phenolic_oh(O,C) :- hydroxy(O,C), aromatic(C), carbon(C), ring_atom(R,C), arene_ring(R).
% enol: OH on a C=C carbon
enol_oh(O,C) :- hydroxy(O,C), carbon(C), double(C,C2), carbon(C2).
% ether: C-O-C, both single, neither carbon a carbonyl (ester/anhydride excluded)
ether_o(O) :- oxygen(O), charge(O,0), single(O,C1), single(O,C2), C1 < C2,
    carbon_or_r(C1), carbon_or_r(C2), not carbonyl_c(C1), not carbonyl_c(C2), heavy_degree(O,2).
% hydroperoxy / peroxide
hydroperoxy(O1,O2) :- hydroxy(O1,O2), oxygen(O2).
peroxide(O1,O2) :- oxygen(O1), oxygen(O2), single(O1,O2), O1 != O2.
% epoxide: 3-membered ring with one O
epoxide_ring(R) :- ring(R,3), ring_atom(R,O), oxygen(O), #count{ A : ring_atom(R,A), carbon(A) } = 2.

% ---------------------------------------------------------------- nitrogen groups
amide_n(C,N) :- carbonyl(C,_), nitrogen(N), sa(C,N).
% carboxamide: R-C(=O)-N (R = C/R/H)
carboxamide(C,N) :- amide_n(C,N), carbonyl(C,O),
    #count{ B : nb(C,B), B != O, B != N, not carbon_or_r(B) } = 0.
% amine nitrogen: neutral sp3 N bonded only to C/H/R (single bonds), not an amide/imide N,
% not attached to a carbonyl/thiocarbonyl/imine carbon or an aromatic ring N
amine_n(N) :- nitrogen(N), charge(N,0), not aromatic(N), not unsat_atom(N),
    #count{ B : nb(N,B), not carbon_or_r(B), heavy(B) } = 0,
    not n_on_acyl(N), not n_on_amidine(N).
n_on_acyl(N) :- nitrogen(N), single(N,C), carbon(C), double(C,X), heteroatom(X).
n_on_amidine(N) :- nitrogen(N), single(N,C), carbon(C), double(C,X), nitrogen(X).
amine_class(N,K) :- amine_n(N), K = #count{ B : nb(N,B), carbon_or_r(B) }.
primary_amine_n(N)   :- amine_class(N,1).
secondary_amine_n(N) :- amine_class(N,2).
tertiary_amine_n(N)  :- amine_class(N,3).
% protonated amines / quaternary ammonium (N+ with only C/H substituents, single bonds)
ammonium_n(N) :- nitrogen(N), charge(N,1), not aromatic(N), not unsat_atom(N),
    #count{ B : nb(N,B), not carbon_or_r(B), heavy(B) } = 0.
quaternary_ammonium_n(N) :- ammonium_n(N), hcount(N,0), #count{ B : nb(N,B), carbon_or_r(B) } = 4.
% amino group attached to C (neutral or protonated, NH2 / NH3+)
amino_n(N,C) :- nitrogen(N), single(N,C), carbon(C), not aromatic(N), heavy_degree(N,1),
    charge(N,0), hcount(N,2).
amino_n(N,C) :- nitrogen(N), single(N,C), carbon(C), not aromatic(N), heavy_degree(N,1),
    charge(N,1), hcount(N,3).
nitrile_c(C) :- carbon(C), triple(C,N), nitrogen(N), heavy_degree(N,1).
nitro_n(N) :- nitrogen(N), #count{ O : nb(N,O), oxygen(O), heavy_degree(O,1) } = 2,
    #count{ X : nb(N,X), heavy(X) } = 3.
imine_c(C,N) :- carbon(C), double(C,N), nitrogen(N).
oxime(C,N,O) :- imine_c(C,N), single(N,O), hydroxy(O,N).
azo(N1,N2) :- nitrogen(N1), nitrogen(N2), double(N1,N2).
urea_c(C) :- carbonyl(C,_), #count{ N : single(C,N), nitrogen(N) } = 2.
carbamate_c(C) :- carbonyl(C,_), single(C,N), nitrogen(N), single(C,O), oxygen(O), not carbonyl(C,O).
guanidine_c(C) :- carbon(C), #count{ N : nb(C,N), nitrogen(N) } = 3, double(C,N1), nitrogen(N1).
isocyanate_n(N) :- nitrogen(N), double(N,C), carbon(C), double(C,O), oxygen(O).
isothiocyanate_n(N) :- nitrogen(N), double(N,C), carbon(C), double(C,S), sulfur(S).

% ---------------------------------------------------------------- sulfur groups
thiol_s(S,C) :- sulfur(S), hcount(S,1), charge(S,0), single(S,C), carbon(C).
sulfide_s(S) :- sulfur(S), charge(S,0), heavy_degree(S,2), single(S,C1), single(S,C2), C1 < C2,
    carbon_or_r(C1), carbon_or_r(C2).
disulfide(S1,S2) :- sulfur(S1), sulfur(S2), single(S1,S2), S1 != S2.
thiocarbonyl(C,S) :- carbon(C), double(C,S), sulfur(S).
% S(=O)(=O) and sulfonic acid / sulfonate: C-S(=O)(=O)-O[H/-]
sulfonyl_s(S) :- sulfur(S), #count{ O : double(S,O), oxygen(O) } = 2.
sulfonic_acid_s(S) :- sulfonyl_s(S), single(S,C), carbon_or_r(C), single(S,O), hydroxy(O,S).
sulfonate_s(S)     :- sulfonyl_s(S), single(S,C), carbon_or_r(C), single(S,O), oxido(O,S).
sulfonamide_s(S)   :- sulfonyl_s(S), single(S,N), nitrogen(N).
sulfoxide_s(S) :- sulfur(S), #count{ O : double(S,O), oxygen(O) } = 1, #count{ C : single(S,C), carbon(C) } = 2.
% sulfate ester / sulfuric acid derivatives: S bonded to 4 O
sulfate_s(S) :- sulfur(S), #count{ O : nb(S,O), oxygen(O) } = 4.

% ---------------------------------------------------------------- phosphorus groups
% phosphate-type P: P bonded to 4 O (phosphoric acid, its esters/anhydrides and anions)
phosphate_p(P) :- phosphorus(P), #count{ O : nb(P,O), oxygen(O) } = 4.
% P-O-P bridge (diphosphate, triphosphate ...)
phosphoanhydride_o(O) :- oxygen(O), nb(O,P1), nb(O,P2), P1 < P2, phosphorus(P1), phosphorus(P2).
% P-O-C ester oxygen
phospho_ester_o(P,O,C) :- phosphorus(P), single(P,O), oxygen(O), single(O,C), carbon_or_r(C).
phosphonate_p(P) :- phosphorus(P), #count{ O : nb(P,O), oxygen(O) } = 3, single(P,C), carbon(C).

% ---------------------------------------------------------------- rings
ring_size_atom(A,N) :- ring_atom(R,A), ring(R,N).
ring_has_el(R,E) :- ring_atom(R,A), element(A,E).
hetero_ring(R) :- ring_atom(R,A), heteroatom(A).
carbocycle(R) :- ring(R,_), not hetero_ring(R).
% arene ring: aromatic ring made of carbons only
arene_ring(R) :- aromatic_ring(R), carbocycle(R).
benzene_ring(R) :- arene_ring(R), ring(R,6).
heteroaromatic_ring(R) :- aromatic_ring(R), hetero_ring(R).
% two SSSR rings sharing at least one bond (ortho-fused, includes bridged)
% (indexed via ring bonds per ring; ~4x cheaper than joining ring_atom 4 times)
rbond_ring(A,B,R) :- ring_atom(R,A), ring_atom(R,B), A < B, ring_bond(A,B).
fused(R1,R2) :- rbond_ring(A,B,R1), rbond_ring(A,B,R2), R1 != R2.
% spiro: share exactly one atom
shares_atom(R1,R2) :- ring_atom(R1,A), ring_atom(R2,A), R1 != R2.
spiro(R1,R2) :- shares_atom(R1,R2), not fused(R1,R2),
    #count{ B : ring_atom(R1,B), ring_atom(R2,B) } = 1.
% ring systems: fused rings belong to the same system (rep = smallest ring id).
% Labels only propagate downwards (X < R2), so the minimum still reaches every ring
% of the system, but far fewer pairs are grounded than with a full closure.
rs_lab(R,R) :- ring(R,_).
rs_lab(R2,X) :- rs_lab(R1,X), fused(R1,R2), X < R2.
ring_system_rep(R,S) :- ring(R,_), S = #min{ X : rs_lab(R,X) }.
same_ring_system(R1,R2) :- ring_system_rep(R1,S), ring_system_rep(R2,S).
ring_system(S) :- ring_system_rep(_,S).
ring_system_size(S,N) :- ring_system(S), N = #count{ R : ring_system_rep(R,S) }.
% atom attached to an arene ring
on_arene(A,C) :- nb(A,C), aromatic(C), carbon(C), ring_atom(R,C), arene_ring(R), not ring_atom(R,A).

% ---------------------------------------------------------------- molecule level
organic(M) :- has_atom(M,A), carbon(A).
inorganic(M) :- mol(M), not organic(M).
has_element(M,E) :- has_atom(M,A), element(A,E).
carbon_count(M,N) :- elem_count(M,c,N).
carbon_count(M,0) :- mol(M), not elem_count(M,c,_).
anion(M)  :- net_charge(M,Q), Q < 0.
cation(M) :- net_charge(M,Q), Q > 0.
neutral(M) :- net_charge(M,0).
has_charged_atom(M) :- has_atom(M,A), charged(A).
zwitterion_like(M) :- net_charge(M,0), has_atom(M,A), charge(A,Q1), Q1 > 0,
    has_atom(M,B), charge(B,Q2), Q2 < 0.
has_rgroup(M) :- has_atom(M,A), rgroup(A).
single_component(M) :- num_components(M,1).
acyclic(M) :- mol(M), num_rings(M,0).
cyclic(M) :- num_rings(M,N), N > 0.
has_aromatic_ring(M) :- ring_of(M,R), aromatic_ring(R).
has_benzene_ring(M) :- ring_of(M,R), benzene_ring(R).
% only C and H (hydrocarbon)
hydrocarbon(M) :- organic(M), #count{ A : has_atom(M,A), element(A,E), E != c, E != h } = 0.

has_carboxylic_acid(M) :- has_atom(M,C), carboxylic_acid_group(C).
has_carboxylate(M)     :- has_atom(M,C), carboxylate_group(C).
has_carboxy(M)         :- has_atom(M,C), carboxy_group(C).
n_carboxylic_acid(M,N) :- mol(M), N = #count{ C : has_atom(M,C), carboxylic_acid_group(C) }.
n_carboxylate(M,N)     :- mol(M), N = #count{ C : has_atom(M,C), carboxylate_group(C) }.
n_carboxy(M,N)         :- mol(M), N = #count{ C : has_atom(M,C), carboxy_group(C) }.
has_carboxylic_ester(M):- has_atom(M,C), carboxylic_ester(C,_,_).
has_lactone(M)         :- has_atom(M,C), lactone(C,_).
has_aldehyde(M)        :- has_atom(M,C), aldehyde_group(C).
has_ketone(M)          :- has_atom(M,C), ketone_group(C).
has_hydroxy(M)         :- has_atom(M,O), hydroxy_o(O).
has_alcohol(M)         :- has_atom(M,O), alcohol_oh(O,_).
has_phenol(M)          :- has_atom(M,O), phenolic_oh(O,_).
has_ether(M)           :- has_atom(M,O), ether_o(O).
has_carboxamide(M)     :- has_atom(M,C), carboxamide(C,_).
has_amine(M)           :- has_atom(M,N), amine_n(N).
has_amino(M)           :- has_atom(M,N), amino_n(N,_).
has_halogen(M)         :- has_atom(M,A), halogen(A).
has_phosphate(M)       :- has_atom(M,P), phosphate_p(P).
has_epoxide(M)         :- ring_of(M,R), epoxide_ring(R).

% ---------------------------------------------------------------- constant unfoldings
% Popper cannot place constants in a rule, so each useful value of a constant-carrying
% argument gets its own predicate.
fluorine(A)   :- element(A,f).
chlorine(A)   :- element(A,cl).
bromine(A)    :- element(A,br).
iodine(A)     :- element(A,i).
boron(A)      :- element(A,b).
silicon(A)    :- element(A,si).
arsenic(A)    :- element(A,as).

charge_m1(A) :- charge(A,-1).
charge_m2(A) :- charge(A,-2).
charge_p1(A) :- charge(A,1).
charge_p2(A) :- charge(A,2).
charge_p3(A) :- charge(A,3).
neg_charged(A) :- charge(A,Q), Q < 0.
pos_charged(A) :- charge(A,Q), Q > 0.

hcount0(A) :- hcount(A,0).
hcount1(A) :- hcount(A,1).
hcount2(A) :- hcount(A,2).
hcount3(A) :- hcount(A,3).
hcount4(A) :- hcount(A,4).

heavy_degree0(A) :- heavy_degree(A,0).
heavy_degree1(A) :- heavy_degree(A,1).
heavy_degree2(A) :- heavy_degree(A,2).
heavy_degree3(A) :- heavy_degree(A,3).
heavy_degree4(A) :- heavy_degree(A,4).

dative(A,B) :- bond(A,B,dative).
bond_e(A,B) :- bond_cip(A,B,e).
bond_z(A,B) :- bond_cip(A,B,z).
cip_r(A) :- cip(A,r).
cip_s(A) :- cip(A,s).
radical_atom(A) :- radical(A,_).
isotope_atom(A) :- isotope(A,_).

ring3(R) :- ring(R,3).
ring4(R) :- ring(R,4).
ring5(R) :- ring(R,5).
ring6(R) :- ring(R,6).
ring7(R) :- ring(R,7).
ring8(R) :- ring(R,8).
macrocycle(R) :- ring(R,N), N >= 12.
in_ring3(A) :- ring_size_atom(A,3).
in_ring4(A) :- ring_size_atom(A,4).
in_ring5(A) :- ring_size_atom(A,5).
in_ring6(A) :- ring_size_atom(A,6).
in_ring7(A) :- ring_size_atom(A,7).
in_ring8(A) :- ring_size_atom(A,8).
fused_ring(R) :- fused(R,_).

steroid_atom(A) :- steroid_pos(A,_).
has_steroid_core(M) :- has_atom(M,A), steroid_pos(A,_).
