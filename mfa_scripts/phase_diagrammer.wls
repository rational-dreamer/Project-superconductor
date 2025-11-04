#!/usr/bin/env wolframscript
(* ::Package:: *)

Off[FindRoot::lstol];
Off[CloudConnect::verr];
Off[FindRoot::cvmit];


(* Functions *)

intersectIndex[a_List,b_List]:=Flatten[Position[Table[If[(a[[i]]-b[[i]])(a[[i+1]]-b[[i+1]])<-10^-8,True,False],{i,1,Min[Length[a],Length[b]]-1}],True]]
intersectArg[ind_,x_List,f_List,g_List]:=(x[[ind+1]]*(f[[ind]]-g[[ind]])+x[[ind]]*(g[[ind+1]]-f[[ind+1]]))/((f[[ind]]-g[[ind]])+(g[[ind+1]]-f[[ind+1]]))
intersectFunc[ind_,f_List,g_List]:=(f[[ind]]*g[[ind+1]]-g[[ind]]*f[[ind+1]])/((f[[ind]]-g[[ind]])+(g[[ind+1]]-f[[ind+1]]))
interpolateFunc[t_,ind_,x_List,f_List]:=f[[ind]]+((f[[ind+1]]-f[[ind]])(t-x[[ind]]))/(x[[ind+1]]-x[[ind]])
zeroboundIndex[a_List]:=Module[{\[Theta],\[Theta]1},Flatten[\[Theta]=Table[If[a[[i]]<10^-6,0,1],{i,1,Length[a]}];\[Theta]1=Table[\[Theta][[i+1]]-\[Theta][[i]],{i,1,Length[a]-1}];{Position[\[Theta]1,1],Position[\[Theta]1,-1]+1}]]
minVal[a_List]:=Min[DeleteCases[a,Null]]
minPos[a_List]:=Flatten[Position[a,minVal[a]]][[1]]
(* Equations of MFs *)



nb[V_,\[Mu]_,hz_]:=(\[Mu]-hz)/(4*V)
aSol[V_,ha_]:=ha/(4*V)
lzSol[J_,gaz_]:=gaz/(4*J*(1/2)^2)
BxSol[tb_,h2x_]:=h2x/(2*tb)


(* T=0 energies *)

fe0CO[V_,\[CapitalDelta]_,n_]:=\[CapitalDelta]-2*V+(4*V-\[CapitalDelta])Abs[n]
fe0AFM[V_,J_,\[CapitalDelta]_,n_]:=2(V-J*(1/2)^2)n^2+(\[CapitalDelta]+4*J*(1/2)^2)Abs[n]-2*J*(1/2)^2
fe0BS[V_,tb_,\[CapitalDelta]_,n_]:=\[CapitalDelta]+2*V*n^2-tb*(1-n^2)

(* NO phase *)

\[Phi][\[CapitalDelta]_,\[Beta]_,n_]=Sqrt[(1-n^2)E^(-2\[Beta]*\[CapitalDelta])+n^2];
frEnNO[V_,\[CapitalDelta]_,\[Beta]_,n_]=-(1/\[Beta])*Log[2*(1+\[Phi][\[CapitalDelta],\[Beta],n])/(1-n^2)]+\[CapitalDelta]*Abs[n]+1/\[Beta]*Abs[n]*Log[(Abs[n]+\[Phi][\[CapitalDelta],\[Beta],n])/(1-Abs[n])]+2V*n^2;

(* CO phase *)

ZCO2[\[CapitalDelta]_,\[Beta]_,hz_,ha_]:=4(1+E^(-\[Beta]*\[CapitalDelta]) Cosh[\[Beta]*(hz+ha)])(1+E^(-\[Beta]*\[CapitalDelta]) Cosh[\[Beta]*(hz-ha)])
\[Omega]CO[V_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,hz_,ha_]:=-(1/(2*\[Beta]))Log[ZCO2[\[CapitalDelta],\[Beta],hz,ha]]-(\[Mu]-hz)^2/(8*V)+ha^2/(8*V)
eqCO[V_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,x_,y_]:=4*V*Sinh[\[Beta]*x]/(E^(\[Beta]*\[CapitalDelta])+Cosh[\[Beta]*x])-\[Mu]+y

(* mf = { hz, ha } *)
Parallelize[
mfCO[V_,\[CapitalDelta]_,T_,\[Mu]_]:={(x+y)/2,(x-y)/2}/.FindRoot[{eqCO[V,\[CapitalDelta],1/T,\[Mu],x,y]==0,eqCO[V,\[CapitalDelta],1/T,\[Mu],y,x]==0},{x,\[Mu]+4*V},{y,\[Mu]-4*V}]
]
(* AFM phase *)

ZAFM[\[CapitalDelta]_,\[Beta]_,hz_,gaz_]:=2(Cosh[\[Beta]*gaz]+E^(-\[Beta]*\[CapitalDelta]) Cosh[\[Beta]*hz])
\[Omega]AFM[V_,J_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,hz_,gaz_]:=-(1/\[Beta])Log[ZAFM[\[CapitalDelta],\[Beta],hz,gaz]]-(\[Mu]-hz)^2/(8*V)+gaz^2/(8*J*(1/2)^2)
eq1AFM[V_,J_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,x_,y_]:=4*V*Sinh[\[Beta]*x]/(E^(\[Beta]*\[CapitalDelta]) Cosh[\[Beta]*y]+Cosh[\[Beta]*x])-\[Mu]+x
eq2AFM[V_,J_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,x_,y_]:=4*J*(1/2)^2*Sinh[\[Beta]*y]/(Cosh[\[Beta]*y]+E^(-\[Beta]*\[CapitalDelta]) Cosh[\[Beta]*x])-y

(* mf = { hz, gaz } *)
Parallelize[
mfAFM[V_,J_,\[CapitalDelta]_,T_,\[Mu]_]:={x,y}/.FindRoot[{eq1AFM[V,J,\[CapitalDelta],1/T,\[Mu],x,y]==0,eq2AFM[V,J,\[CapitalDelta],1/T,\[Mu],x,y]==0},{x,\[Mu]},{y,4*J*(1/2)^2}]
]
(* BS phase *)

ZBS[\[CapitalDelta]_,\[Beta]_,hz_,h2x_]:=2(1+E^(-\[Beta]*\[CapitalDelta]) Cosh[\[Beta]*Sqrt[hz^2+h2x^2]])
\[Omega]BS[V_,tb_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,hz_,h2x_]:=-(1/\[Beta])Log[ZBS[\[CapitalDelta],\[Beta],hz,h2x]]-(\[Mu]-hz)^2/(8*V)+h2x^2/(4*tb)
eq1BS[V_,tb_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,x_,y_]:=4*V*x/Sqrt[x^2+y^2]*Sinh[\[Beta]*Sqrt[x^2+y^2]]/(E^(\[Beta]*\[CapitalDelta])+Cosh[\[Beta]*Sqrt[x^2+y^2]])-\[Mu]+x
eq2BS[V_,tb_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,x_,y_]:=2*tb*y/Sqrt[x^2+y^2]*Sinh[\[Beta]*Sqrt[x^2+y^2]]/(E^(\[Beta]*\[CapitalDelta])+Cosh[\[Beta]*Sqrt[x^2+y^2]])-y

(* mf = { hz, h2x } *)
Parallelize[
mfBS[V_,tb_,\[CapitalDelta]_,T_,\[Mu]_]:={x,y}/.FindRoot[{eq1BS[V,tb,\[CapitalDelta],1/T,\[Mu],x,y]==0,eq2BS[V,tb,\[CapitalDelta],1/T,\[Mu],x,y]==0},{x,\[Mu]-(4*V)/(1+E^(\[CapitalDelta]/T))},{y,2*tb}]
]
(* FL phase *)

zcFL[\[CapitalDelta]_,\[Beta]_,hz_,hp_,hn_]=1+E^(-(2/3) \[Beta] (\[CapitalDelta]+Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]]))+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]))+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]));
nFL[\[CapitalDelta]_,\[Beta]_,hz_,hp_,hn_]=(-E^(-(2/3) \[Beta] (\[CapitalDelta]+Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]])) (18 hz Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)] Cos[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]]+27 (3 hn^4-3 hp^4-2 hz^3 \[CapitalDelta]+2 hz \[CapitalDelta]^3+hp^2 (6 hz^2+7 hz \[CapitalDelta]-\[CapitalDelta]^2)+hn^2 (-6 hz^2+7 hz \[CapitalDelta]+\[CapitalDelta]^2)) Sin[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]])+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])])) (18 hz Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)] Cos[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]+27 (3 hn^4-3 hp^4-2 hz^3 \[CapitalDelta]+2 hz \[CapitalDelta]^3+hp^2 (6 hz^2+7 hz \[CapitalDelta]-\[CapitalDelta]^2)+hn^2 (-6 hz^2+7 hz \[CapitalDelta]+\[CapitalDelta]^2)) Sin[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])])+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])])) (18 hz Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)] Cos[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]+27 (3 hn^4-3 hp^4-2 hz^3 \[CapitalDelta]+2 hz \[CapitalDelta]^3+hp^2 (6 hz^2+7 hz \[CapitalDelta]-\[CapitalDelta]^2)+hn^2 (-6 hz^2+7 hz \[CapitalDelta]+\[CapitalDelta]^2)) Sin[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]))/(9 (1+E^(-(2/3) \[Beta] (\[CapitalDelta]+Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]]))+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]))+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]))) Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)]);
PFL[\[CapitalDelta]_,\[Beta]_,hz_,hp_,hn_]=(hp (E^(-(2/3) \[Beta] (\[CapitalDelta]+Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]])) (-2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)] Cos[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]]+3 (hn^2 (15 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-hp^2+2 hz (hz+\[CapitalDelta]))) Sin[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]])+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])])) (2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)] Cos[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]-3 (hn^2 (15 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-hp^2+2 hz (hz+\[CapitalDelta]))) Sin[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])])+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])])) (2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)] Cos[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]-3 (hn^2 (15 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-hp^2+2 hz (hz+\[CapitalDelta]))) Sin[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])])))/((1+E^(-(2/3) \[Beta] (\[CapitalDelta]+Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]]))+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]))+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]))) Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)]);
NFL[\[CapitalDelta]_,\[Beta]_,hz_,hp_,hn_]=(hn (E^(-(2/3) \[Beta] (\[CapitalDelta]+Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]])) (-2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)] Cos[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]]+3 (hn^2 (3 hz-\[CapitalDelta])-hp^2 (15 hz+\[CapitalDelta])-2 hz (3 hz^2-4 hz \[CapitalDelta]+\[CapitalDelta]^2)) Sin[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]])+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])])) (2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)] Cos[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]-3 (hn^2 (3 hz-\[CapitalDelta])-hp^2 (15 hz+\[CapitalDelta])-2 hz (3 hz^2-4 hz \[CapitalDelta]+\[CapitalDelta]^2)) Sin[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])])+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])])) (2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)] Cos[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]-3 (hn^2 (3 hz-\[CapitalDelta])-hp^2 (15 hz+\[CapitalDelta])-2 hz (3 hz^2-4 hz \[CapitalDelta]+\[CapitalDelta]^2)) Sin[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])])))/((1+E^(-(2/3) \[Beta] (\[CapitalDelta]+Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])]]))+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (-\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]))+E^(-(2/3) \[Beta] (\[CapitalDelta]-Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Cos[1/3 (\[Pi]+ArcCos[(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))/(2 Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3])])]))) Sqrt[3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2] Sqrt[(3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3] Sqrt[1-(9 hn^2 (3 hz-\[CapitalDelta])+(3 hz+\[CapitalDelta]) (-9 hp^2+6 hz \[CapitalDelta]-2 \[CapitalDelta]^2))^2/(4 (3 hn^2+3 hp^2+3 hz^2+\[CapitalDelta]^2)^3)]);
\[Omega]FL[V_,tp_,tn_,tpn_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,hz_,hp_,hn_]:=-(1/\[Beta])Log[zcFL[\[CapitalDelta],\[Beta],hz,hp,hn]]-(\[Mu]-hz)^2/(8*V)+tp*PFL[\[CapitalDelta],\[Beta],hz,hp,hn]^2+tn*NFL[\[CapitalDelta],\[Beta],hz,hp,hn]^2+tpn*PFL[\[CapitalDelta],\[Beta],hz,hp,hn]*NFL[\[CapitalDelta],\[Beta],hz,hp,hn]

eqn1FL[V_,tp_,tn_,tpn_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,z_,x_,y_]:=4*V*nFL[\[CapitalDelta],\[Beta],z,x,y]-\[Mu]+z
eqn2FL[V_,tp_,tn_,tpn_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,z_,x_,y_]:=2tp*PFL[\[CapitalDelta],\[Beta],z,x,y]+tpn*NFL[\[CapitalDelta],\[Beta],z,x,y]-x
eqn3FL[V_,tp_,tn_,tpn_,\[CapitalDelta]_,\[Beta]_,\[Mu]_,z_,x_,y_]:=tpn*PFL[\[CapitalDelta],\[Beta],z,x,y]+2tn*NFL[\[CapitalDelta],\[Beta],z,x,y]-y

(* mf = { hz, hp, hn } *)
Parallelize[
mfFL[V_,tp_,tn_,tpn_,\[CapitalDelta]_,T_,\[Mu]_]:={z,x,y}/.FindRoot[{eqn1FL[V,tp,tn,tpn,\[CapitalDelta],1/T,\[Mu],z,x,y]==0,eqn2FL[V,tp,tn,tpn,\[CapitalDelta],1/T,\[Mu],z,x,y]==0,eqn3FL[V,tp,tn,tpn,\[CapitalDelta],1/T,\[Mu],z,x,y]==0},{z,\[Mu]-V},{x,2*tp},{y,2*tn}]
]


(* Functions *)
\[CapitalDelta]t=Rationalize[ToExpression[$ScriptCommandLine[[2]]]];
Vt=Rationalize[ToExpression[$ScriptCommandLine[[3]]]];
Jt=Rationalize[ToExpression[$ScriptCommandLine[[4]]]];
tbt=Rationalize[ToExpression[$ScriptCommandLine[[5]]]];
tpt=Rationalize[ToExpression[$ScriptCommandLine[[6]]]];
tnt=Rationalize[ToExpression[$ScriptCommandLine[[7]]]];
tpnt=Rationalize[ToExpression[$ScriptCommandLine[[8]]]];
neps=10^-5;



\[Mu]min=-3.; \[Mu]max=3.; \[Mu]num=200;
\[Mu]S=Table[\[Mu]min+((\[Mu]max-\[Mu]min)/(\[Mu]num-1))(i-1),{i,1,\[Mu]num}];
Tmin=0.01; Tmax=0.3; Tnum=100;
dT=(Tmax-Tmin)/(Tnum-1);
TS=Table[Tmin+((Tmax-Tmin)/(Tnum-1))(i-1),{i,1,Tnum}];
nmin=-0.99;nmax=0.99;nnum=99;
nS=Table[nmin+((nmax-nmin)/(nnum-1))(i-1),{i,1,nnum}];



(*CO*)
DistributeDefinitions[\[Mu]S[[i]],TS[[j]],TS[[j]],Tnum,\[Mu]num];
mfCOsurfHT=ParallelTable[{\[Mu]S[[i]],TS[[j]],mfCO[Vt,\[CapitalDelta]t,TS[[j]],\[Mu]S[[i]]]},{j,1,Tnum},{i,1,\[Mu]num}];
hzCOsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],mfCOsurfHT[[j,i,3,1]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
nbCOsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],nb[Vt,\[Mu]S[[i]],mfCOsurfHT[[j,i,3,1]]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
haCOsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],mfCOsurfHT[[j,i,3,2]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
haCOsurfNT=ParallelTable[{nbCOsurfHT[[k,3]],nbCOsurfHT[[k,2]],haCOsurfHT[[k,3]]},{k,1,\[Mu]num*Tnum}];
aCOsurfNT=ParallelTable[{nbCOsurfHT[[k,3]],nbCOsurfHT[[k,2]],aSol[Vt,haCOsurfHT[[k,3]]]},{k,1,\[Mu]num*Tnum}];
\[Omega]COsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],\[Omega]CO[Vt,\[CapitalDelta]t, 1/TS[[j]], \[Mu]S[[i]], mfCOsurfHT[[j,i,3,1]],mfCOsurfHT[[j,i,3,2]]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
\[Omega]COlinesHT=ParallelTable[{\[Mu]S[[i]],\[Omega]CO[Vt,\[CapitalDelta]t,1/TS[[j]],\[Mu]S[[i]],mfCOsurfHT[[j,i,3,1]],mfCOsurfHT[[j,i,3,2]]]},{j,1,Tnum},{i,1,\[Mu]num}];
nbCOlinesHT=ParallelTable[{\[Mu]S[[i]],nb[Vt,\[Mu]S[[i]],mfCOsurfHT[[j,i,3,1]]]},{j,1,Tnum},{i,1,\[Mu]num}];
hzCOlinesHT=ParallelTable[{\[Mu]S[[i]],mfCOsurfHT[[j,i,3,1]]},{j,1,Tnum},{i,1,\[Mu]num}];
haCOlinesHT=ParallelTable[{\[Mu]S[[i]],mfCOsurfHT[[j,i,3,2]]},{j,1,Tnum},{i,1,\[Mu]num}];
feCOsurfNT=ParallelTable[{nbCOsurfHT[[k,3]],nbCOsurfHT[[k,2]],\[Omega]COsurfHT[[k,3]]+nbCOsurfHT[[k,1]]*nbCOsurfHT[[k,3]]},{k,1,\[Mu]num*Tnum}];
feCOsurfNT1=ParallelTable[{nbCOsurfHT[[k,3]],nbCOsurfHT[[k,2]],-(\[Omega]COsurfHT[[k,3]]+nbCOsurfHT[[k,1]]*nbCOsurfHT[[k,3]])},{k,1,\[Mu]num*Tnum}];
feCOlinesNT=ParallelTable[{nbCOlinesHT[[j,i,2]],\[Omega]CO[Vt,\[CapitalDelta]t,1/TS[[j]],\[Mu]S[[i]],mfCOsurfHT[[j,i,3,1]],mfCOsurfHT[[j,i,3,2]]]+\[Mu]S[[i]]*nbCOlinesHT[[j,i,2]]},{j,1,Tnum},{i,1,\[Mu]num}];
DistributeDefinitions[aL,bL]; 
Module[{aL,bL},
Table[
aL=feCOlinesNT[[j]];
bL=DeleteCases[Prepend[ParallelTable[If[aL[[i+1,1]]-aL[[i,1]]>neps,aL[[i+1]]],{i,1,\[Mu]num-1}],aL[[1]]],Null];
feCOfuncNT[j]=Interpolation[bL,InterpolationOrder->1]
,{j,1,Tnum}]
];



(* AFM *)
DistributeDefinitions[\[Mu]S[[i]],TS[[j]],TS[[j]],Tnum,\[Mu]num]; 
(*SetSharedVariable[mfAFMsurfHT,hzAFMsurfHT,nbAFMsurfHT,gazAFMsurfHT,gazAFMsurfNT,lzAFMsurfNT,\[Omega]AFMsurfHT,\[Omega]AFMlinesHT,nbAFMlinesHT,hzAFMlinesHT,gazAFMlinesHT,feAFMsurfNT,feAFMsurfNT1,feAFMlinesNT,mfAFM,Vt,Jt,\[CapitalDelta]t,TS,\[Mu]S];*)
mfAFMsurfHT=ParallelTable[{\[Mu]S[[i]],TS[[j]],mfAFM[Vt,Jt,\[CapitalDelta]t,TS[[j]],\[Mu]S[[i]]]},{j,1,Tnum},{i,1,\[Mu]num}];
hzAFMsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],mfAFMsurfHT[[j,i,3,1]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
nbAFMsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],nb[Vt,\[Mu]S[[i]],mfAFMsurfHT[[j,i,3,1]]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
gazAFMsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],mfAFMsurfHT[[j,i,3,2]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
gazAFMsurfNT=ParallelTable[{nbAFMsurfHT[[k,3]],nbAFMsurfHT[[k,2]],gazAFMsurfHT[[k,3]]},{k,1,\[Mu]num*Tnum}];
lzAFMsurfNT=ParallelTable[{nbAFMsurfHT[[k,3]],nbAFMsurfHT[[k,2]],lzSol[Jt,gazAFMsurfHT[[k,3]]]},{k,1,\[Mu]num*Tnum}];
\[Omega]AFMsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],\[Omega]AFM[Vt,Jt,\[CapitalDelta]t,1/TS[[j]],\[Mu]S[[i]],mfAFMsurfHT[[j,i,3,1]],mfAFMsurfHT[[j,i,3,2]]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
\[Omega]AFMlinesHT=ParallelTable[{\[Mu]S[[i]],\[Omega]AFM[Vt,Jt,\[CapitalDelta]t,1/TS[[j]],\[Mu]S[[i]],mfAFMsurfHT[[j,i,3,1]],mfAFMsurfHT[[j,i,3,2]]]},{j,1,Tnum},{i,1,\[Mu]num}];
nbAFMlinesHT=ParallelTable[{\[Mu]S[[i]],nb[Vt,\[Mu]S[[i]],mfAFMsurfHT[[j,i,3,1]]]},{j,1,Tnum},{i,1,\[Mu]num}];
hzAFMlinesHT=ParallelTable[{\[Mu]S[[i]],mfAFMsurfHT[[j,i,3,1]]},{j,1,Tnum},{i,1,\[Mu]num}];
gazAFMlinesHT=ParallelTable[{\[Mu]S[[i]],mfAFMsurfHT[[j,i,3,2]]},{j,1,Tnum},{i,1,\[Mu]num}];
feAFMsurfNT=ParallelTable[{nbAFMsurfHT[[k,3]],nbAFMsurfHT[[k,2]],\[Omega]AFMsurfHT[[k,3]]+nbAFMsurfHT[[k,1]]*nbAFMsurfHT[[k,3]]},{k,1,\[Mu]num*Tnum}];
feAFMsurfNT1=ParallelTable[{nbAFMsurfHT[[k,3]],nbAFMsurfHT[[k,2]],-(\[Omega]AFMsurfHT[[k,3]]+nbAFMsurfHT[[k,1]]*nbAFMsurfHT[[k,3]])},{k,1,\[Mu]num*Tnum}];
feAFMlinesNT=ParallelTable[{nbAFMlinesHT[[j,i,2]],\[Omega]AFM[Vt,Jt,\[CapitalDelta]t,1/TS[[j]],\[Mu]S[[i]],mfAFMsurfHT[[j,i,3,1]],mfAFMsurfHT[[j,i,3,2]]]+\[Mu]S[[i]]*nbAFMlinesHT[[j,i,2]]},{j,1,Tnum},{i,1,\[Mu]num}];

DistributeDefinitions[aL,bL]; 
Module[{aL,bL},
Table[
aL=feAFMlinesNT[[j]];
bL=DeleteCases[Prepend[ParallelTable[If[aL[[i+1,1]]-aL[[i,1]]>neps,aL[[i+1]]],{i,1,\[Mu]num-1}],aL[[1]]],Null];
feAFMfuncNT[j]=Interpolation[bL,InterpolationOrder->1]
,{j,1,Tnum}]
];



(* BS *)
DistributeDefinitions[\[Mu]S[[i]],TS[[j]],TS[[j]],Tnum,\[Mu]num]; 
(*SetSharedVariable[\[Mu]S,nb,TS,mfBS,Vt,tbt,\[CapitalDelta]t,TS,\[Mu]S,mfBSsurfHT,hzBSsurfHT,nbBSsurfHT,h2xBSsurfHT,h2xBSsurfNT,BxBSsurfNT,\[Omega]BSsurfHT,\[Omega]BSlinesHT,nbBSlinesHT,hzBSlinesHT,h2xBSlinesHT,feBSsurfNT,feBSsurfNT1,feBSlinesNT];*)
mfBSsurfHT=ParallelTable[{\[Mu]S[[i]],TS[[j]],mfBS[Vt,tbt,\[CapitalDelta]t,TS[[j]],\[Mu]S[[i]]]},{j,1,Tnum},{i,1,\[Mu]num}];
hzBSsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],mfBSsurfHT[[j,i,3,1]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
nbBSsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],nb[Vt,\[Mu]S[[i]],mfBSsurfHT[[j,i,3,1]]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
h2xBSsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],mfBSsurfHT[[j,i,3,2]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
h2xBSsurfNT=ParallelTable[{nbBSsurfHT[[k,3]],nbBSsurfHT[[k,2]],h2xBSsurfHT[[k,3]]},{k,1,\[Mu]num*Tnum}];
BxBSsurfNT=ParallelTable[{nbBSsurfHT[[k,3]],nbBSsurfHT[[k,2]],BxSol[tbt,h2xBSsurfHT[[k,3]]]},{k,1,\[Mu]num*Tnum}];
\[Omega]BSsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],\[Omega]BS[Vt,tbt,\[CapitalDelta]t,1/TS[[j]],\[Mu]S[[i]],mfBSsurfHT[[j,i,3,1]],mfBSsurfHT[[j,i,3,2]]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
\[Omega]BSlinesHT=ParallelTable[{\[Mu]S[[i]],\[Omega]BS[Vt,tbt,\[CapitalDelta]t,1/TS[[j]],\[Mu]S[[i]],mfBSsurfHT[[j,i,3,1]],mfBSsurfHT[[j,i,3,2]]]},{j,1,Tnum},{i,1,\[Mu]num}];
nbBSlinesHT=ParallelTable[{\[Mu]S[[i]],nb[Vt,\[Mu]S[[i]],mfBSsurfHT[[j,i,3,1]]]},{j,1,Tnum},{i,1,\[Mu]num}];
hzBSlinesHT=ParallelTable[{\[Mu]S[[i]],mfBSsurfHT[[j,i,3,1]]},{j,1,Tnum},{i,1,\[Mu]num}];
h2xBSlinesHT=ParallelTable[{\[Mu]S[[i]],mfBSsurfHT[[j,i,3,2]]},{j,1,Tnum},{i,1,\[Mu]num}];
feBSsurfNT=ParallelTable[{nbBSsurfHT[[k,3]],nbBSsurfHT[[k,2]],\[Omega]BSsurfHT[[k,3]]+nbBSsurfHT[[k,1]]*nbBSsurfHT[[k,3]]},{k,1,\[Mu]num*Tnum}];
feBSsurfNT1=ParallelTable[{nbBSsurfHT[[k,3]],nbBSsurfHT[[k,2]],-(\[Omega]BSsurfHT[[k,3]]+nbBSsurfHT[[k,1]]*nbBSsurfHT[[k,3]])},{k,1,\[Mu]num*Tnum}];
feBSlinesNT=ParallelTable[{nbBSlinesHT[[j,i,2]],\[Omega]BS[Vt,tbt,\[CapitalDelta]t,1/TS[[j]],\[Mu]S[[i]],mfBSsurfHT[[j,i,3,1]],mfBSsurfHT[[j,i,3,2]]]+\[Mu]S[[i]]*nbBSlinesHT[[j,i,2]]},{j,1,Tnum},{i,1,\[Mu]num}];
DistributeDefinitions[aL,bL]; 
Module[{aL,bL},
Table[
aL=feBSlinesNT[[j]];
bL=DeleteCases[Prepend[ParallelTable[If[aL[[i+1,1]]-aL[[i,1]]>neps,aL[[i+1]]],{i,1,\[Mu]num-1}],aL[[1]]],Null];
feBSfuncNT[j]=Interpolation[bL,InterpolationOrder->1]
,{j,1,Tnum}]
];


(*--------------------------------------------------------------------------------------*)
(* FL *)
DistributeDefinitions[\[Mu]S[[i]],TS[[j]],TS[[j]],Tnum,\[Mu]num]; 

mfFLsurfHT=ParallelTable[{\[Mu]S[[i]],TS[[j]],mfFL[Vt,tpt,tnt,tpnt,\[CapitalDelta]t,TS[[j]],\[Mu]S[[i]]]},{j,1,Tnum},{i,1,\[Mu]num}];
hzFLsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],mfFLsurfHT[[j,i,3,1]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
nbFLsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],nb[Vt,\[Mu]S[[i]],mfFLsurfHT[[j,i,3,1]]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
hpFLsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],mfFLsurfHT[[j,i,3,2]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
hpFLsurfNT=ParallelTable[{nbFLsurfHT[[k,3]],nbFLsurfHT[[k,2]],hpFLsurfHT[[k,3]]},{k,1,\[Mu]num*Tnum}];
hnFLsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],mfFLsurfHT[[j,i,3,3]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
hnFLsurfNT=ParallelTable[{nbFLsurfHT[[k,3]],nbFLsurfHT[[k,2]],hnFLsurfHT[[k,3]]},{k,1,\[Mu]num*Tnum}];
PFLsurfNT=ParallelTable[{nbFLsurfHT[[k,3]],nbFLsurfHT[[k,2]],PFL[\[CapitalDelta]t,1/nbFLsurfHT[[k,2]],hzFLsurfHT[[k,3]],hpFLsurfHT[[k,3]],hnFLsurfHT[[k,3]]]},{k,1,\[Mu]num*Tnum}];
NFLsurfNT=ParallelTable[{nbFLsurfHT[[k,3]],nbFLsurfHT[[k,2]],NFL[\[CapitalDelta]t,1/nbFLsurfHT[[k,2]],hzFLsurfHT[[k,3]],hpFLsurfHT[[k,3]],hnFLsurfHT[[k,3]]]},{k,1,\[Mu]num*Tnum}];
\[Omega]FLsurfHT=Flatten[ParallelTable[{\[Mu]S[[i]],TS[[j]],\[Omega]FL[Vt,tpt,tnt,tpnt,\[CapitalDelta]t,1/TS[[j]],\[Mu]S[[i]],mfFLsurfHT[[j,i,3,1]],mfFLsurfHT[[j,i,3,2]],mfFLsurfHT[[j,i,3,3]]]},{j,1,Tnum},{i,1,\[Mu]num}],1];
\[Omega]FLlinesHT=ParallelTable[{\[Mu]S[[i]],\[Omega]FL[Vt,tpt,tnt,tpnt,\[CapitalDelta]t,1/TS[[j]],\[Mu]S[[i]],mfFLsurfHT[[j,i,3,1]],mfFLsurfHT[[j,i,3,2]],mfFLsurfHT[[j,i,3,3]]]},{j,1,Tnum},{i,1,\[Mu]num}];
nbFLlinesHT=ParallelTable[{\[Mu]S[[i]],nb[Vt,\[Mu]S[[i]],mfFLsurfHT[[j,i,3,1]]]},{j,1,Tnum},{i,1,\[Mu]num}];
hzFLlinesHT=ParallelTable[{\[Mu]S[[i]],mfFLsurfHT[[j,i,3,1]]},{j,1,Tnum},{i,1,\[Mu]num}];
hpFLlinesHT=ParallelTable[{\[Mu]S[[i]],mfFLsurfHT[[j,i,3,2]]},{j,1,Tnum},{i,1,\[Mu]num}];
hnFLlinesHT=ParallelTable[{\[Mu]S[[i]],mfFLsurfHT[[j,i,3,3]]},{j,1,Tnum},{i,1,\[Mu]num}];
feFLsurfNT=ParallelTable[{nbFLsurfHT[[k,3]],nbFLsurfHT[[k,2]],\[Omega]FLsurfHT[[k,3]]+nbFLsurfHT[[k,1]]*nbFLsurfHT[[k,3]]},{k,1,\[Mu]num*Tnum}];
feFLsurfNT1=ParallelTable[{nbFLsurfHT[[k,3]],nbFLsurfHT[[k,2]],-(\[Omega]FLsurfHT[[k,3]]+nbFLsurfHT[[k,1]]*nbFLsurfHT[[k,3]])},{k,1,\[Mu]num*Tnum}];
feFLlinesNT=ParallelTable[{nbFLlinesHT[[j,i,2]],\[Omega]FL[Vt,tpt,tnt,tpnt,\[CapitalDelta]t,1/TS[[j]],\[Mu]S[[i]],mfFLsurfHT[[j,i,3,1]],mfFLsurfHT[[j,i,3,2]],mfFLsurfHT[[j,i,3,3]]]+\[Mu]S[[i]]*nbFLlinesHT[[j,i,2]]},{j,1,Tnum},{i,1,\[Mu]num}];
DistributeDefinitions[aL,bL]; 

Module[{aL,bL},
Table[
aL=feFLlinesNT[[j]];
bL=DeleteCases[Prepend[ParallelTable[If[aL[[i+1,1]]-aL[[i,1]]>neps,aL[[i+1]]],{i,1,\[Mu]num-1}],aL[[1]]],Null];
feFLfuncNT[j]=Interpolation[bL,InterpolationOrder->1]
,{j,1,Tnum}]
];



(* Phase separation *)
(*ParallelCombine[Table[f[5^100 + i], {i, #}] &, Range[0, 10], Join] *)
(*Table[i0COAFM[t]=intersectIndex[\[Omega]COlinesHT[[t]]\[Transpose][[2]],\[Omega]AFMlinesHT[[t]]\[Transpose][[2]]];\[Mu]0COAFM[t]=Table[intersectArg[i,\[Mu]S,\[Omega]COlinesHT[[t]]\[Transpose][[2]],\[Omega]AFMlinesHT[[t]]\[Transpose][[2]]],{i,i0COAFM[t]}];
\[Omega]0COAFM[t]=ParallelTable[intersectFunc[i,\[Omega]COlinesHT[[t]]\[Transpose][[2]],\[Omega]AFMlinesHT[[t]]\[Transpose][[2]]],{i,i0COAFM[t]}];

nbCOAFM[t]=Table[{interpolateFunc[\[Mu]0COAFM[t][[i]],i0COAFM[t][[i]],\[Mu]S,nbCOlinesHT[[t]]\[Transpose][[2]]],interpolateFunc[\[Mu]0COAFM[t][[i]],i0COAFM[t][[i]],\[Mu]S,nbAFMlinesHT[[t]]\[Transpose][[2]]]},{i,Length[i0COAFM[t]]}],{t,1,Tnum}];
psCOAFM=Flatten[ParallelTable[If[nbCOAFM[t]!={},Table[{{nbCOAFM[t][[i,1]],TS[[t]]},{nbCOAFM[t][[i,2]],TS[[t]]}},{i,Length[i0COAFM[t]]}]],{t,1,Tnum}],2];

FEpsCOAFMfuncS[i_,t_,n_]:=If[\[Mu]0COAFM[t]!={}&&Min[nbCOAFM[t][[i]]]<=n<=Max[nbCOAFM[t][[i]]],\[Omega]0COAFM[t][[i]]+\[Mu]0COAFM[t][[i]]*n]
percCOAFM[i_,t_,n_]:=If[\[Mu]0COAFM[t]!={}&&Min[nbCOAFM[t][[i]]]<=n<=Max[nbCOAFM[t][[i]]],{(n-nbCOAFM[t][[i,2]])/(nbCOAFM[t][[i,1]]-nbCOAFM[t][[i,2]]),(nbCOAFM[t][[i,1]]-n)/(nbCOAFM[t][[i,1]]-nbCOAFM[t][[i,2]])}];

Table[i0COBS[t]=intersectIndex[\[Omega]COlinesHT[[t]]\[Transpose][[2]],\[Omega]BSlinesHT[[t]]\[Transpose][[2]]];\[Mu]0COBS[t]=Table[intersectArg[i,\[Mu]S,\[Omega]COlinesHT[[t]]\[Transpose][[2]],\[Omega]BSlinesHT[[t]]\[Transpose][[2]]],{i,i0COBS[t]}];
\[Omega]0COBS[t]=Table[intersectFunc[i,\[Omega]COlinesHT[[t]]\[Transpose][[2]],\[Omega]BSlinesHT[[t]]\[Transpose][[2]]],{i,i0COBS[t]}];
nbCOBS[t]=Table[{interpolateFunc[\[Mu]0COBS[t][[i]],i0COBS[t][[i]],\[Mu]S,nbCOlinesHT[[t]]\[Transpose][[2]]],interpolateFunc[\[Mu]0COBS[t][[i]],i0COBS[t][[i]],\[Mu]S,nbBSlinesHT[[t]]\[Transpose][[2]]]},{i,Length[i0COBS[t]]}],{t,1,Tnum}];
psCOBS=Flatten[ParallelTable[If[nbCOBS[t]!={},Table[{{nbCOBS[t][[i,1]],TS[[t]]},{nbCOBS[t][[i,2]],TS[[t]]}},{i,Length[i0COBS[t]]}]],{t,1,Tnum}],2];

FEpsCOBSfuncS[i_,t_,n_]:=If[\[Mu]0COBS[t]!={}&&Min[nbCOBS[t][[i]]]<=n<=Max[nbCOBS[t][[i]]],\[Omega]0COBS[t][[i]]+\[Mu]0COBS[t][[i]]*n]
percCOBS[i_,t_,n_]:=If[\[Mu]0COBS[t]!={}&&Min[nbCOBS[t][[i]]]<=n<=Max[nbCOBS[t][[i]]],{(n-nbCOBS[t][[i,2]])/(nbCOBS[t][[i,1]]-nbCOBS[t][[i,2]]),(nbCOBS[t][[i,1]]-n)/(nbCOBS[t][[i,1]]-nbCOBS[t][[i,2]])}];

Table[i0AFMBS[t]=intersectIndex[\[Omega]AFMlinesHT[[t]]\[Transpose][[2]],\[Omega]BSlinesHT[[t]]\[Transpose][[2]]];\[Mu]0AFMBS[t]=Table[intersectArg[i,\[Mu]S,\[Omega]AFMlinesHT[[t]]\[Transpose][[2]],\[Omega]BSlinesHT[[t]]\[Transpose][[2]]],{i,i0AFMBS[t]}];
\[Omega]0AFMBS[t]=Table[intersectFunc[i,\[Omega]AFMlinesHT[[t]]\[Transpose][[2]],\[Omega]BSlinesHT[[t]]\[Transpose][[2]]],{i,i0AFMBS[t]}];
nbAFMBS[t]=Table[{interpolateFunc[\[Mu]0AFMBS[t][[i]],i0AFMBS[t][[i]],\[Mu]S,nbAFMlinesHT[[t]]\[Transpose][[2]]],interpolateFunc[\[Mu]0AFMBS[t][[i]],i0AFMBS[t][[i]],\[Mu]S,nbBSlinesHT[[t]]\[Transpose][[2]]]},{i,Length[i0AFMBS[t]]}],{t,1,Tnum}];
psAFMBS=Flatten[ParallelTable[If[nbAFMBS[t]!={},Table[{{nbAFMBS[t][[i,1]],TS[[t]]},{nbAFMBS[t][[i,2]],TS[[t]]}},{i,Length[i0AFMBS[t]]}]],{t,1,Tnum}],2];

FEpsAFMBSfuncS[i_,t_,n_]:=If[\[Mu]0AFMBS[t]!={}&&Min[nbAFMBS[t][[i]]]<=n<=Max[nbAFMBS[t][[i]]],\[Omega]0AFMBS[t][[i]]+\[Mu]0AFMBS[t][[i]]*n]
percAFMBS[i_,t_,n_]:=If[\[Mu]0AFMBS[t]!={}&&Min[nbAFMBS[t][[i]]]<=n<=Max[nbAFMBS[t][[i]]],{(n-nbAFMBS[t][[i,2]])/(nbAFMBS[t][[i,1]]-nbAFMBS[t][[i,2]]),(nbAFMBS[t][[i,1]]-n)/(nbAFMBS[t][[i,1]]-nbAFMBS[t][[i,2]])}];

Table[i0FLCO[t]=intersectIndex[\[Omega]FLlinesHT[[t]]\[Transpose][[2]],\[Omega]COlinesHT[[t]]\[Transpose][[2]]];\[Mu]0FLCO[t]=Table[intersectArg[i,\[Mu]S,\[Omega]FLlinesHT[[t]]\[Transpose][[2]],\[Omega]COlinesHT[[t]]\[Transpose][[2]]],{i,i0FLCO[t]}];
\[Omega]0FLCO[t]=Table[intersectFunc[i,\[Omega]FLlinesHT[[t]]\[Transpose][[2]],\[Omega]COlinesHT[[t]]\[Transpose][[2]]],{i,i0FLCO[t]}];
nbFLCO[t]=Table[{interpolateFunc[\[Mu]0FLCO[t][[i]],i0FLCO[t][[i]],\[Mu]S,nbFLlinesHT[[t]]\[Transpose][[2]]],interpolateFunc[\[Mu]0FLCO[t][[i]],i0FLCO[t][[i]],\[Mu]S,nbCOlinesHT[[t]]\[Transpose][[2]]]},{i,Length[i0FLCO[t]]}],{t,1,Tnum}];
psFLCO=Flatten[ParallelTable[If[nbFLCO[t]!={},Table[{{nbFLCO[t][[i,1]],TS[[t]]},{nbFLCO[t][[i,2]],TS[[t]]}},{i,Length[i0FLCO[t]]}]],{t,1,Tnum}],2];

FEpsFLCOfuncS[i_,t_,n_]:=If[\[Mu]0FLCO[t]!={}&&Min[nbFLCO[t][[i]]]<=n<=Max[nbFLCO[t][[i]]],\[Omega]0FLCO[t][[i]]+\[Mu]0FLCO[t][[i]]*n]
percFLCO[i_,t_,n_]:=If[\[Mu]0FLCO[t]!={}&&Min[nbFLCO[t][[i]]]<=n<=Max[nbFLCO[t][[i]]],{(n-nbFLCO[t][[i,2]])/(nbFLCO[t][[i,1]]-nbFLCO[t][[i,2]]),(nbFLCO[t][[i,1]]-n)/(nbFLCO[t][[i,1]]-nbFLCO[t][[i,2]])}];

Table[i0FLAFM[t]=intersectIndex[\[Omega]FLlinesHT[[t]]\[Transpose][[2]],\[Omega]AFMlinesHT[[t]]\[Transpose][[2]]];\[Mu]0FLAFM[t]=Table[intersectArg[i,\[Mu]S,\[Omega]FLlinesHT[[t]]\[Transpose][[2]],\[Omega]AFMlinesHT[[t]]\[Transpose][[2]]],{i,i0FLAFM[t]}];
\[Omega]0FLAFM[t]=Table[intersectFunc[i,\[Omega]FLlinesHT[[t]]\[Transpose][[2]],\[Omega]AFMlinesHT[[t]]\[Transpose][[2]]],{i,i0FLAFM[t]}];
nbFLAFM[t]=Table[{interpolateFunc[\[Mu]0FLAFM[t][[i]],i0FLAFM[t][[i]],\[Mu]S,nbFLlinesHT[[t]]\[Transpose][[2]]],interpolateFunc[\[Mu]0FLAFM[t][[i]],i0FLAFM[t][[i]],\[Mu]S,nbAFMlinesHT[[t]]\[Transpose][[2]]]},{i,Length[i0FLAFM[t]]}],{t,1,Tnum}];
psFLAFM=Flatten[Table[If[nbFLAFM[t]!={},Table[{{nbFLAFM[t][[i,1]],TS[[t]]},{nbFLAFM[t][[i,2]],TS[[t]]}},{i,Length[i0FLAFM[t]]}]],{t,1,Tnum}],2];

FEpsFLAFMfuncS[i_,t_,n_]:=If[\[Mu]0FLAFM[t]!={}&&Min[nbFLAFM[t][[i]]]<=n<=Max[nbFLAFM[t][[i]]],\[Omega]0FLAFM[t][[i]]+\[Mu]0FLAFM[t][[i]]*n]
percFLAFM[i_,t_,n_]:=If[\[Mu]0FLAFM[t]!={}&&Min[nbFLAFM[t][[i]]]<=n<=Max[nbFLAFM[t][[i]]],{(n-nbFLAFM[t][[i,2]])/(nbFLAFM[t][[i,1]]-nbFLAFM[t][[i,2]]),(nbFLAFM[t][[i,1]]-n)/(nbFLAFM[t][[i,1]]-nbFLAFM[t][[i,2]])}];

Table[i0FLBS[t]=intersectIndex[\[Omega]FLlinesHT[[t]]\[Transpose][[2]],\[Omega]BSlinesHT[[t]]\[Transpose][[2]]];\[Mu]0FLBS[t]=Table[intersectArg[i,\[Mu]S,\[Omega]FLlinesHT[[t]]\[Transpose][[2]],\[Omega]BSlinesHT[[t]]\[Transpose][[2]]],{i,i0FLBS[t]}];
\[Omega]0FLBS[t]=Table[intersectFunc[i,\[Omega]FLlinesHT[[t]]\[Transpose][[2]],\[Omega]BSlinesHT[[t]]\[Transpose][[2]]],{i,i0FLBS[t]}];
nbFLBS[t]=Table[{interpolateFunc[\[Mu]0FLBS[t][[i]],i0FLBS[t][[i]],\[Mu]S,nbFLlinesHT[[t]]\[Transpose][[2]]],interpolateFunc[\[Mu]0FLBS[t][[i]],i0FLBS[t][[i]],\[Mu]S,nbBSlinesHT[[t]]\[Transpose][[2]]]},{i,Length[i0FLBS[t]]}],{t,1,Tnum}];
psFLBS=Flatten[Table[If[nbFLBS[t]!={},Table[{{nbFLBS[t][[i,1]],TS[[t]]},{nbFLBS[t][[i,2]],TS[[t]]}},{i,Length[i0FLBS[t]]}]],{t,1,Tnum}],2];

FEpsFLBSfuncS[i_,t_,n_]:=If[\[Mu]0FLBS[t]!={}&&Min[nbFLBS[t][[i]]]<=n<=Max[nbFLBS[t][[i]]],\[Omega]0FLBS[t][[i]]+\[Mu]0FLBS[t][[i]]*n]
percFLBS[i_,t_,n_]:=If[\[Mu]0FLBS[t]!={}&&Min[nbFLBS[t][[i]]]<=n<=Max[nbFLBS[t][[i]]],{(n-nbFLBS[t][[i,2]])/(nbFLBS[t][[i,1]]-nbFLBS[t][[i,2]]),(nbFLBS[t][[i,1]]-n)/(nbFLBS[t][[i,1]]-nbFLBS[t][[i,2]])}]
*)
fePureS[t_,n_]:={feCOfuncNT[t][n],feAFMfuncNT[t][n],feBSfuncNT[t][n],feFLfuncNT[t][n],frEnNO[Vt,\[CapitalDelta]t,1/TS[[t]],n]}
feS[t_,n_]:={feCOfuncNT[t][n],feAFMfuncNT[t][n],feBSfuncNT[t][n],feFLfuncNT[t][n],FEpsCOAFMfuncS[1,t,n],FEpsCOAFMfuncS[2,t,n],FEpsCOBSfuncS[1,t,n],FEpsCOBSfuncS[2,t,n],FEpsAFMBSfuncS[1,t,n],FEpsAFMBSfuncS[2,t,n],FEpsFLCOfuncS[1,t,n],FEpsFLCOfuncS[2,t,n],FEpsFLAFMfuncS[1,t,n],FEpsFLAFMfuncS[2,t,n],FEpsFLBSfuncS[1,t,n],FEpsFLBSfuncS[2,t,n],frEnNO[Vt,\[CapitalDelta]t,1/TS[[t]],n]}
colorCO=Blue;
colorAFM=Red;
colorBS=Green;
colorFL=Gray;
colorNO=LightGray;
colorS[1,t_,n_]:=colorCO
colorS[2,t_,n_]:=colorAFM
colorS[3,t_,n_]:=colorBS
colorS[4,t_,n_]:=colorFL
colorS[5,t_,n_]:=Blend[{colorCO,colorAFM},percCOAFM[1,t,n]]
colorS[6,t_,n_]:=Blend[{colorCO,colorAFM},percCOAFM[2,t,n]]
colorS[7,t_,n_]:=Blend[{colorCO,colorBS},percCOBS[1,t,n]]
colorS[8,t_,n_]:=Blend[{colorCO,colorBS},percCOBS[2,t,n]]
colorS[9,t_,n_]:=Blend[{colorAFM,colorBS},percAFMBS[1,t,n]]
colorS[10,t_,n_]:=Blend[{colorAFM,colorBS},percAFMBS[2,t,n]]
colorS[11,t_,n_]:=Blend[{colorFL,colorCO},percFLCO[1,t,n]]
colorS[12,t_,n_]:=Blend[{colorFL,colorCO},percFLCO[2,t,n]]
colorS[13,t_,n_]:=Blend[{colorFL,colorAFM},percFLAFM[1,t,n]]
colorS[14,t_,n_]:=Blend[{colorFL,colorAFM},percFLAFM[2,t,n]]
colorS[15,t_,n_]:=Blend[{colorFL,colorBS},percFLBS[1,t,n]]
colorS[16,t_,n_]:=Blend[{colorFL,colorBS},percFLBS[2,t,n]]
colorS[17,t_,n_]:=colorNO

CONOcomp[t_,n_]:=Block[{mPos=minPos[{feCOfuncNT[t][n],frEnNO[Vt,\[CapitalDelta]t,1/TS[[t]],n]}],phsS={0,1}},
If[mPos==1,phsS={1,0}];
phsS]
AFMNOcomp[t_,n_]:=Block[{mPos=minPos[{feAFMfuncNT[t][n],frEnNO[Vt,\[CapitalDelta]t,1/TS[[t]],n]}],phsS={0,1}},
If[mPos==1,phsS={1,0}];
phsS]
BSNOcomp[t_,n_]:=Block[{mPos=minPos[{feBSfuncNT[t][n],frEnNO[Vt,\[CapitalDelta]t,1/TS[[t]],n]}],phsS={0,1}},
If[mPos==1,phsS={1,0}];
phsS]
FLNOcomp[t_,n_]:=Block[{mPos=minPos[{feFLfuncNT[t][n],frEnNO[Vt,\[CapitalDelta]t,1/TS[[t]],n]}],phsS={0,1}},
If[mPos==1,phsS={1,0}];
phsS]

phasePureComp[t_,n_]:=Block[{mPos=minPos[fePureS[t,n]],phsS={0,0,0,0,1}},
If[mPos==1,phsS={1,0,0,0,0}];If[mPos==2,phsS={0,1,0,0,0}];If[mPos==3,phsS={0,0,1,0,0}];If[mPos==4,phsS={0,0,0,1,0}];If[mPos==5,phsS={0,0,0,0,1}];
phsS]

phaseComp[t_,n_]:=Block[{mPos=minPos[feS[t,n]],phsS={0,0,0,0,1}},
If[mPos==1,phsS={1,0,0,0,0}];If[mPos==2,phsS={0,1,0,0,0}];If[mPos==3,phsS={0,0,1,0,0}];If[mPos==4,phsS={0,0,0,1,0}];If[mPos==5,phsS={percCOAFM[1,t,n][[1]],percCOAFM[1,t,n][[2]],0,0,0}];If[mPos==6,phsS={percCOAFM[2,t,n][[1]],percCOAFM[2,t,n][[2]],0,0,0}];If[mPos==7,phsS={percCOBS[1,t,n][[1]],0,percCOBS[1,t,n][[2]],0,0}];If[mPos==8,phsS={percCOBS[2,t,n][[1]],0,percCOBS[2,t,n][[2]],0,0}];If[mPos==9,phsS={0,percAFMBS[1,t,n][[1]],percAFMBS[1,t,n][[2]],0,0}];If[mPos==10,phsS={0,percAFMBS[2,t,n][[1]],percAFMBS[2,t,n][[2]],0,0}];If[mPos==11,phsS={percFLCO[1,t,n][[2]],0,0,percFLCO[1,t,n][[1]],0}];If[mPos==12,phsS={percFLCO[2,t,n][[2]],0,0,percFLCO[2,t,n][[1]],0}];If[mPos==13,phsS={0,percFLAFM[1,t,n][[2]],0,percFLAFM[1,t,n][[1]],0}];If[mPos==14,phsS={0,percFLAFM[2,t,n][[2]],0,percFLAFM[2,t,n][[1]],0}];If[mPos==15,phsS={0,0,percFLBS[1,t,n][[2]],percFLBS[1,t,n][[1]],0}];If[mPos==16,phsS={0,0,percFLBS[2,t,n][[2]],percFLBS[2,t,n][[1]],0}];If[mPos==17,phsS={0,0,0,0,1}];
phsS]

phaseFE[t_,n_]:=minVal[feS[t,n]]

colorCO=Blue;
colorAFM=Red;
colorBS=Green;
colorFL=Gray;
colorNO=LightGray;
colS[phComp_]:=Blend[{colorCO,colorAFM,colorBS,colorFL,colorNO},phComp]
rec[{n_,T_},{dn_,dT_}]:=Rectangle[{n-dn/2,T-dT/2},{n+dn/2,T+dT/2}]



nMin=0.;nMax=0.8;dn=0.005;
tilePure=Table[{colS[phasePureComp[t,n]],rec[{n,TS[[t]]},{dn,dT}]},{t,1,Tnum},{n,nMin,nMax,dn}];
d05=Graphics[tilePure,Frame->True,AspectRatio->1]
graph = d05



(*nMin=0.;nMax=0.75;dn=0.005;
tile=Table[{colS[phaseComp[t,n]],rec[{n,TS[[t]]},{dn,dT}]},{t,1,Tnum},{n,nMin,nMax,dn}];fCO=Interpolation[Flatten[Table[Table[{n,TS[[t]],phaseComp[t,n][[1]]},{n,nMin,nMax,dn}],{t,1,Tnum}],1]];
fAFM=Interpolation[Flatten[Table[Table[{n,TS[[t]],phaseComp[t,n][[2]]},{n,nMin,nMax,dn}],{t,1,Tnum}],1]];
fBS=Interpolation[Flatten[Table[Table[{n,TS[[t]],phaseComp[t,n][[3]]},{n,nMin,nMax,dn}],{t,1,Tnum}],1]];
fFL=Interpolation[Flatten[Table[Table[{n,TS[[t]],phaseComp[t,n][[4]]},{n,nMin,nMax,dn}],{t,1,Tnum}],1]];
c05=ContourPlot[{fCO[n,T]==0.5,fAFM[n,T]==0.5,fBS[n,T]==0.5,fFL[n,T]==0.5},{n,nMin,nMax},{T,Tmin,Tmax},ContourStyle->Black]
d05=Graphics[tile,Frame->True,AspectRatio->1]
Show[d05,c05]
graph=Show[d05,c05]*)


(*dir=SetDirectory[""];*)

\[CapitalDelta]t=NumberForm[\[CapitalDelta]t,{4,4}];
Vt=NumberForm[Vt,{4,4}];
tbt=NumberForm[tbt,{4,4}];
tpt=NumberForm[tpt,{4,4}];
tnt=NumberForm[tnt,{4,4}];
tpnt=NumberForm[tpnt,{4,4}];
Jt=NumberForm[Jt,{4,4}];
filenameTemplate=StringTemplate["L32_D`\[CapitalDelta]t`_V`Vt`_tb`tbt`_tp`tpt`_tn`tnt`_tpn`tpnt`_J`Jt`.png"];
filename=filenameTemplate[<|"\[CapitalDelta]t"->\[CapitalDelta]t|>,<|"Vt"->Vt|>,<|"tbt"->tbt|>,<|"tpt"->tpt|>,<|"tnt"->tnt|>,<|"tpnt"->tpnt|>,<|"Jt"->Jt|>];

(*Export[FileNameJoin[{dir,filename}],graph,ImageSize->600];*)
Export[FileNameJoin[filename],graph,ImageSize->600];


