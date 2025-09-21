import streamlit as st
import pandas as pd
from pickle import load
import joblib
import pickle
import numpy as np
import math
import os
from config.definitions import ROOT_DIR
from PIL import Image
import matplotlib.pyplot as plt

#Load model and scaler
NGBoost_LSC=joblib.load(os.path.join(ROOT_DIR,'Studs_Deck_LSC_NGBoost.joblib'))
NGBoost_LSC_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Deck_LSC_NGBoost.pkl'),'rb'))

CatBoost_LSC=joblib.load(os.path.join(ROOT_DIR,'Studs_Deck_LSC_CatBoost.joblib'))
CatBoost_LSC_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Deck_LSC_CatBoost.pkl'),'rb'))

st.write('### Load-Slip Curves for Headed Shear Studs in Deck Slab Ribs Perpendicular to Supports Predicted by ML Models')

st.sidebar.header('User input parameters')

units=st.sidebar.radio('Units',('SI','US customary units'))

if units=='SI':
    fcm = st.sidebar.number_input('Concrete compressive strength, $f_\mathrm{cm}$ (MPa)', min_value=23, max_value=65, value=38)
   
    SW = st.sidebar.selectbox('Stud welding type, $SW$', ('Through-deck','Through-holes'), index=0)
    if SW=='Through-deck': d=19
    elif SW=='Through-holes': d=st.sidebar.selectbox("Stud shank diameter, $d$ (mm)",(19,22))
    nr = st.sidebar.selectbox('Number of studs in a rib, $n_\mathrm{r}$', (1,2), index=0)
    
    if nr==2: SP=st.sidebar.selectbox("Stud position",('Two studs in parallel','Two studs in series','Two staggered studs'))
    else: SP="N/A"
    
    if SP=="Two studs in parallel": sy_min, sy_max, sy_step, sx=(math.ceil(3.95*d/1)*1.0, math.floor(7.68*d/1)*1.0, 1.0, 0.0)
    elif SP=="Two studs in series": sy, sx_min, sx_max, sx_step=(0.0, math.ceil(2.79*d/1)*1.0, math.floor(4.66*d/1)*1.0, 1.0)
    elif SP=="Two staggered studs": sy_min, sy_max, sy_step, sx_min, sx_max, sx_step=(math.ceil(3.75*d/1)*1.0, math.floor(5.25*d/1)*1.0, 1.0, math.ceil(2.79*d/1)*1.0, math.floor(4.67*d/1)*1.0, 1.0)

    if SP=="Two studs in parallel": sy=st.sidebar.number_input("Transverse stud spacing in a rib, $s_\mathrm{y}$ (mm)",min_value=sy_min, max_value=sy_max, step=sy_step, format="%.0f")
    elif SP=="Two studs in series": sx=st.sidebar.number_input("Longitudinal stud spacing in a rib, $s_\mathrm{x}$ (mm)",min_value=sx_min, max_value=sx_max, step=sx_step, format="%.0f") 
    elif SP=="Two staggered studs": sy, sx=(st.sidebar.number_input("Transverse stud spacing in a rib, $s_\mathrm{y}$ (mm)",min_value=sy_min, max_value=sy_max, step=sy_step, format="%.0f"), st.sidebar.number_input("Longitudinal stud spacing in a rib, $s_\mathrm{x}$ (mm)",min_value=sx_min, max_value=sx_max, step=sx_step, format="%.0f"))  
    if nr==1: sy, sx=(0.0, 0.0)    
    
    hpn = st.sidebar.number_input('Deck height, $h_\mathrm{pn}$ (mm)', min_value=38, max_value=83, value=51)
    bbot = st.sidebar.number_input('Concrete rib width measured at the deck bottom, $b_\mathrm{bot}$ (mm)', min_value=math.ceil(max(62.0,sx+d)/1)*1.0, max_value=136.0, value=math.ceil(max(62.0,sx+d)/1)*1.0)
    btop = st.sidebar.number_input('Concrete rib width measured at the deck top, $b_\mathrm{top}$ (mm)', min_value=math.ceil(max(bbot/0.82,101)/1)*1.0, max_value=math.floor(min(bbot/0.50,240.0)/1)*1.0, value=math.ceil(max(bbot/0.82,101)/1)*1.0)
    t = st.sidebar.number_input('Deck thickness, $t$ (mm)', min_value=math.ceil(max(0.009*hpn,0.75)/0.01)*0.01, max_value=math.floor(min(0.0237*hpn,1.20)/0.01)*0.01, value=math.ceil(max(0.009*hpn,0.75)/0.01)*0.01)
    hsc = st.sidebar.number_input('Stud height, $h_\mathrm{sc}$ (mm)', min_value=math.ceil(max(4.0*d,1.58*d+hpn)/1)*1.0, max_value=math.floor(min(7.9*d,4.21*d+hpn)/1)*1.0, value=math.ceil(max(4.0*d,1.58*d+hpn)/1)*1.0)
    et = st.sidebar.number_input('Longitudinal distance from the rib top corner to the nearest stud center, $e_\mathrm{t}$ (mm)', min_value=math.ceil(max(0.46*hpn,31,(btop-bbot+d)/2)/1)*1.0, max_value=math.floor(min(2.64*hpn,151,(btop+bbot-d)/2)/1)*1.0, value=math.ceil(max(0.46*hpn,31,(btop-bbot+d)/2)/1)*1.0)
    fu = st.sidebar.number_input('Stud tensile strength, $f_\mathrm{u}$ (MPa)', min_value=400.0, max_value=500.0, value=450.0)
    NnormP = st.sidebar.number_input('Normal force on the face of the slab as a percentage of longitudinal force, $N_\mathrm{norm}$ (%)', min_value=0.0, max_value=20.0, value=12.0)
    b0 = (btop+bbot)/2
    e=et-(btop-bbot)/4    
       
    if nr==1 and e==b0/2: Stud_Pos='Central'
    elif nr==1 and e<b0/2: Stud_Pos='Unfavorable'
    elif nr==1 and e>b0/2: Stud_Pos='Favorable'
    elif nr==2 and sx>0 and sy>0: Stud_Pos='Staggered'
    elif nr==2 and sx==0 and sy>0: Stud_Pos='Two studs in parallel'
    elif nr==2 and sx>0 and sy==0: Stud_Pos='Two studs in series'
    
    fcm_psi=fcm/0.00689476
    d_in=d/25.4
    hpn_in=hpn/25.4
    t_in=t/25.4
    hsc_in=hsc/25.4
    fu_ksi=fu/6.89476
    et_in=et/25.4
    b0_in=b0/25.4
    e_in=e/25.4
    btop_in=btop/25.4
    bbot_in=bbot/25.4
    sy_in=sy/25.4
    sx_in=sx/25.4
    Nnorm=NnormP/100
              
else:
    fcm_psi = st.sidebar.number_input('Concrete compressive strength, $f^\prime_\mathrm{c}$ (psi)', min_value=3350, max_value=9400, value=5000)

    SW = st.sidebar.selectbox('Stud welding type, $SW$', ('Through-deck','Through-holes'), index=0)
    if SW=='Through-deck': d_in=0.75
    elif SW=='Through-holes': d_in=st.sidebar.selectbox("Stud shank diameter, $d$ (in.)",(0.75,0.875))
    nr = st.sidebar.selectbox('Number of studs in a rib, $n_\mathrm{r}$', (1,2), index=0)

    if nr==2: SP=st.sidebar.selectbox("Stud position",('Two studs in parallel','Two studs in series','Two staggered studs'))
    else: SP="N/A"
    
    if SP=="Two studs in parallel": sy_min, sy_max, sy_step, sx=(math.ceil(3.95*d_in/1)*1.0, math.floor(7.68*d_in/1)*1.0, 1.0, 0.0)
    elif SP=="Two studs in series": sy, sx_min, sx_max, sx_step=(0.0, math.ceil(2.79*d_in/1)*1.0, math.floor(4.66*d_in/1)*1.0, 1.0)
    elif SP=="Two staggered studs": sy_min, sy_max, sy_step, sx_min, sx_max, sx_step=(math.ceil(3.75*d_in/1)*1.0, math.floor(5.25*d_in/1)*1.0, 1.0, math.ceil(2.79*d_in/1)*1.0, math.floor(4.67*d_in/1)*1.0, 1.0)

    if SP=="Two studs in parallel": sy_in=st.sidebar.number_input("Transverse stud spacing in a rib, $s_\mathrm{y}$ (in.)",min_value=sy_min, max_value=sy_max, step=sy_step, format="%.0f")
    elif SP=="Two studs in series": sx_in=st.sidebar.number_input("Longitudinal stud spacing in a rib, $s_\mathrm{x}$ (in.)",min_value=sx_min, max_value=sx_max, step=sx_step, format="%.0f") 
    elif SP=="Two staggered studs": sy_in, sx_in=(st.sidebar.number_input("Transverse stud spacing in a rib, $s_\mathrm{y}$ (in.)",min_value=sy_min, max_value=sy_max, step=sy_step, format="%.0f"), st.sidebar.number_input("Longitudinal stud spacing in a rib, $s_\mathrm{x}$ (in.)",min_value=sx_min, max_value=sx_max, step=sx_step, format="%.0f"))  
    if nr==1: sy_in, sx_in=(0.0, 0.0)   

    hpn_in = st.sidebar.number_input('Specify deck height, $h_\mathrm{pn}$ (in.)', min_value=1.5, max_value=3.25, value=2.0)
    bbot_in = st.sidebar.number_input('Concrete rib width measured at the deck bottom, $b_\mathrm{bot}$ (in.)', min_value=math.ceil(max(2.44,sx_in+d_in)/0.125)*0.125, max_value=5.35, value=math.ceil(max(2.44,sx_in+d_in)/0.125)*0.125)
    btop_in = st.sidebar.number_input('Concrete rib width measured at the deck top, $b_\mathrm{top}$ (in.)', min_value=math.ceil(max(bbot_in/0.82,3.98)/0.125)*0.125, max_value=math.floor(min(bbot_in/0.50,9.45)/0.125)*0.125, value=math.ceil(max(bbot_in/0.82,3.98)/0.125)*0.125)
    t_in = st.sidebar.number_input('Deck thickness, $t$ (in.)', min_value=math.ceil(max(0.009*hpn_in,0.0295)/0.0001)*0.0001, max_value=math.floor(min(0.0237*hpn_in,0.0474)/0.0001)*0.0001, value=math.ceil(max(0.009*hpn_in,0.0295)/0.0001)*0.0001)
    hsc_in = st.sidebar.number_input('Stud height, $h_\mathrm{sc}$ (in.)', min_value=math.ceil(max(4.0*d_in,1.58*d_in+hpn_in)/0.125)*0.125, max_value=math.floor(min(7.9*d_in,4.21*d_in+hpn_in)/0.125)*0.125, value=math.ceil(max(4.0*d_in,1.58*d_in+hpn_in)/0.125)*0.125)
    et_in = st.sidebar.number_input('Longitudinal distance from the rib top corner to the nearest stud center, $e_\mathrm{t}$ (in.)', min_value=math.ceil(max(0.46*hpn_in,1.22,(btop_in-bbot_in+d_in)/2)/0.125)*0.125, max_value=math.floor(min(2.64*hpn_in,5.94,(btop_in+bbot_in-d_in)/2)/0.125)*0.125, value=math.ceil(max(0.46*hpn_in,1.22,(btop_in-bbot_in+d_in)/2)/0.125)*0.125)
    fu_ksi = st.sidebar.number_input('Stud tensile strength, $f_\mathrm{u}$ (ksi)', min_value=58.0, max_value=72.0, value=65.0)
    NnormP = st.sidebar.number_input('Normal force on the face of the slab as a percentage of longitudinal force, $N_\mathrm{norm}$ (%)', min_value=0.0, max_value=20.0, value=12.0)
    b0_in = (btop_in+bbot_in)/2   
    e_in=et_in-(btop_in-bbot_in)/4    

    if nr==1 and e_in==b0_in/2: Stud_Pos='Central'
    elif nr==1 and e_in<b0_in/2: Stud_Pos='Weak'
    elif nr==1 and e_in>b0_in/2: Stud_Pos='Strong'
    elif nr==2 and sx_in>0 and sy_in>0: Stud_Pos='Staggered'
    elif nr==2 and sx_in==0 and sy_in>0: Stud_Pos='Two studs in parallel'
    elif nr==2 and sx_in>0 and sy_in==0: Stud_Pos='Two studs in series' 

    
    hpn=hpn_in*25.4
    fcm=fcm_psi*0.00689476
    d=d_in*25.4
    hsc=hsc_in*25.4
    fu=fu_ksi*6.89476
    et=et_in*25.4
    sy=sy_in*25.4
    b0=b0_in*25.4
    e=e_in*25.4
    btop=btop_in*25.4
    bbot=bbot_in*25.4
    t=t_in*25.4
    Nnorm=NnormP/100

#Applicability limits
if round(d,0)==19 and hpn<=3*25.4 and b0>=2*25.4 and (hsc-hpn)>=1.5*d and nr==1: AISC_AL='Yes'
elif round(d,0)==19 and hpn<=3*25.4 and b0>=2*25.4 and (hsc-hpn)>=1.5*d and nr==2 and sy==0 and sx>=4*d: AISC_AL='Yes'
elif round(d,0)==19 and hpn<=3*25.4 and b0>=2*25.4 and (hsc-hpn)>=1.5*d and nr==2 and sx==0 and sy>=4*d: AISC_AL='Yes'
elif round(d,0)==19 and hpn<=3*25.4 and b0>=2*25.4 and (hsc-hpn)>=1.5*d and nr==2 and sx>=4*d and sy>=4*d: AISC_AL='Yes'
else: AISC_AL='No'

if (hsc-hpn)>=2.7*d and e>60 and b0>=50 and nr==1: EC4_AL='Yes'
elif (hsc-hpn)>=2.7*d and e>60 and b0>=50 and nr==2 and sy==0 and sx>=5*d: EC4_AL='Yes'
elif (hsc-hpn)>=2.7*d and e>60 and b0>=50 and nr==2 and sx==0 and sy>=4*d: EC4_AL='Yes'
elif (hsc-hpn)>=2.7*d and e>60 and b0>=50 and nr==2 and sx>=5*d and sy>=4*d: EC4_AL='Yes'
elif (hsc-hpn)>=2.0*d and (hsc-hpn)<=2.7*d and e>=25 and e<=60 and fcm<=58 and nr==1 and e==0.5*b0: EC4_AL='Yes'
elif (hsc-hpn)>=2.0*d and (hsc-hpn)<=2.7*d and e>=25 and e<=60 and fcm<=58 and nr==2 and sx==0 and sy>=4*d and e==0.5*b0: EC4_AL='Yes'
elif (hsc-hpn)>=2.0*d and (hsc-hpn)<=2.7*d and e>=25 and e<=60 and fcm<=58 and nr==2 and sx>=5*d and sy>=4*d: EC4_AL='Yes'
else: EC4_AL='No' 


st.write('##### Geometric parameters of welded stud connections')
image1 = Image.open(os.path.join(ROOT_DIR,'Geom_R1.png'))
st.image(image1,width=400)

st.write('##### Positions of two studs in concrete ribs')
image2 = Image.open(os.path.join(ROOT_DIR,'Two_stud_positions.png'))
st.image(image2)

    
Slip1=np.array([0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10,10.5,11.0,11.5,12.0,12.5,13.0,13.5,14.0,14.5,15.0,15.5,16.0,16.5,17.0,17.5,18.0,18.5,19.0,19.5,20.0,20.5,21.0,21.5,22.0,22.5,23.0,23.5]).reshape(-1,1) 

fcm1=np.full((len(Slip1),1),fcm)
d1=np.full((len(Slip1),1),d)
hsc1=np.full((len(Slip1),1),hsc)
et1=np.full((len(Slip1),1),et)
sy1=np.full((len(Slip1),1),sy)
Nnorm1=np.full((len(Slip1),1),Nnorm)

rho_pcf=145
rho=rho_pcf*16.018463

Ecm_AISC=(rho_pcf**1.5)*(fcm_psi/1000)**0.5
Ecm_EC2=22000*(fcm/10)**0.3

if nr==1 and t<=1.0 and SW=='Through-deck' and d<=20: ktmax_EC4=0.85
elif nr==1 and t<=1.0 and SW=='Through-holes' and d>=19: ktmax_EC4=0.75
elif nr==1 and t>1.0 and SW=='Through-deck' and d<=20: ktmax_EC4=1.0
elif nr==1 and t>1.0 and SW=='Through-holes' and d>=19: ktmax_EC4=0.75
elif nr==2 and t<=1.0 and SW=='Through-deck' and d<=20: ktmax_EC4=0.70
elif nr==2 and t<=1.0 and SW=='Through-holes' and d>=19: ktmax_EC4=0.60
elif nr==2 and t>1.0 and SW=='Through-deck' and d<=20: ktmax_EC4=0.80
elif nr==2 and t>1.0 and SW=='Through-holes' and d>=19: ktmax_EC4=0.60
else: ktmax_EC4=0       

kt_EC4=min((0.7/(nr**0.5))*(b0/hpn)*((hsc/hpn)-1),ktmax_EC4)
Pn_EC4_S8=min(0.8*1.1*fu*3.14*(d**2)/4,0.29*(d**2)*(fcm*Ecm_EC2)**0.5)*kt_EC4

C2=min(max(1.85*hpn/b0,1),1.35)

if (e==0.5*b0 or (sy>0 and sx>0)) and SW=='Through-holes': ku=1.0
elif (e==0.5*b0 or (sy>0 and sx>0)) and SW=='Through-deck' and t<1.0: ku=1.0
elif (e==0.5*b0 or (sy>0 and sx>0)) and SW=='Through-deck' and t>=1.0: ku=1.25
else: ku=0

if fcm<=58: fctm=0.3*(fcm-8)**(2/3)
else: fctm=2.12*math.log(1+fcm/10)
fctk=0.7*fctm

Wsc=(2.4*hsc+(nr-1)*sy)*(btop**2)/6
Mplscn=1.1*fu*(d**3)/6
Mplsck=np.where(fu>450,450,fu)*(d**3)/6

if nr==1 or (sy>0 and sx>0): ny=2.0
else: ny=min(1+((hsc-hpn)-2*d)/(0.52*d),2)

Pn_EC4_AG=min(C2*ku*(fctm*Wsc/(hpn*nr)+ny*Mplscn/(0.82*hpn-d/2)),0.58*1.1*fu*3.14*(d**2)/4)

if ((hsc-hpn)>=2.0*d and (hsc-hpn)<2.7*d) or (e>=25 and e<=60): Pn_EC4=Pn_EC4_AG
elif (hsc-hpn)>=2.7*d and e>60: Pn_EC4=Pn_EC4_S8
else: Pn_EC4=0 

Pn_EC4_kips=Pn_EC4/(4.4482216*1000)

PRk_EC4_S8=min(0.8*np.where(fu>450,450,fu)*3.14*(d**2)/4,0.29*(d**2)*((fcm-8)*Ecm_EC2)**0.5)*kt_EC4
PRk_EC4_AG=min(C2*ku*(fctk*Wsc/(hpn*nr)+ny*Mplsck/(0.82*hpn-d/2)),0.58*np.where(fu>450,450,fu)*3.14*(d**2)/4)

if ((hsc-hpn)>=2.0*d and (hsc-hpn)<2.7*d) or (e>=25 and e<=60): PRk_EC4=PRk_EC4_AG
elif (hsc-hpn)>=2.7*d and e>60: PRk_EC4=PRk_EC4_S8
else: PRk_EC4=0

PRk_EC4_kips=PRk_EC4/(4.4482216*1000)


emid_ht=e-d/2
if nr==1: Rg_AISC=1.0
elif nr==2: Rg_AISC=0.85

if emid_ht>=50: Rp_AISC=0.75
else: Rp_AISC=0.60

Pn_AISC_kips=min(0.5*(3.14*(d_in**2)/4)*(fcm_psi*Ecm_AISC)**0.5,Rg_AISC*Rp_AISC*(3.14*(d_in**2)/4)*1.1*fu_ksi)
Pn_AISC=Pn_AISC_kips*(4.4482216*1000)

if nr==1: ksx,ksy=(1.0,1.0)
else: 
    ksx=1.0+0.07*sx/d
    ksy=1.42+0.03*sy/d
Pn_DH=1.17*0.25*3.14*(d**2)*(fcm**0.1)*((1.1*fu)**0.9)*((et/hpn)**0.31)*((bbot/btop)**0.38)*((hsc/d)**0.51)*((t/hpn)**0.30)*ksx*ksy/nr
Pn_DH_kips=Pn_DH/(4.4482216*1000)

PRk_DH=0.9*0.25*3.14*(d**2)*(fcm**0.1)*(fu**0.9)*((et/hpn)**0.31)*((bbot/btop)**0.38)*((hsc/d)**0.51)*((t/hpn)**0.30)*ksx*ksy/nr
PRk_DH_kips=PRk_DH/(4.4482216*1000)

PRk_DH_mod=1.0*0.25*3.14*(d**2)*(fcm**0.1)*(fu**0.9)*((et/hpn)**0.31)*((bbot/btop)**0.38)*((hsc/d)**0.51)*((t/hpn)**0.30)*ksx*ksy/nr
PRk_DH_mod_kips=PRk_DH_mod/(4.4482216*1000)

X_ML_LSC=np.concatenate((Nnorm1,sy1,et1,fcm1,d1,hsc1,Slip1),axis=1)

X_ML_LSC_CatBoost=CatBoost_LSC_scaler.transform(X_ML_LSC)
PoPn_CatBoost = np.concatenate((np.array([0]),CatBoost_LSC.predict(X_ML_LSC_CatBoost)))

X_ML_LSC_NGBoost=NGBoost_LSC_scaler.transform(X_ML_LSC)
PoPn_NGBoost = np.concatenate((np.array([0]),NGBoost_LSC.predict(X_ML_LSC_NGBoost)))

PoPn_NGBoost_dist=NGBoost_LSC.pred_dist(X_ML_LSC_NGBoost)

Predictions_68p3lower = np.concatenate((np.array([0]),PoPn_NGBoost_dist.dist.interval(0.683)[0]))
Predictions_95p4lower = np.concatenate((np.array([0]),PoPn_NGBoost_dist.dist.interval(0.954)[0]))
Predictions_99p7lower = np.concatenate((np.array([0]),PoPn_NGBoost_dist.dist.interval(0.997)[0]))

Predictions_68p3upper = np.concatenate((np.array([0]),PoPn_NGBoost_dist.dist.interval(0.683)[1]))
Predictions_95p4upper = np.concatenate((np.array([0]),PoPn_NGBoost_dist.dist.interval(0.954)[1]))
Predictions_99p7upper = np.concatenate((np.array([0]),PoPn_NGBoost_dist.dist.interval(0.997)[1]))

Slip=np.concatenate((np.array([[0]]),Slip1))

Slip_CatBoost_asc=Slip[0:PoPn_CatBoost.argmax()+1]
PoPn_CatBoost_asc=PoPn_CatBoost[0:PoPn_CatBoost.argmax()+1]
if PoPn_CatBoost.argmax()<PoPn_CatBoost.shape[0]-1:
    Slip_CatBoost_des=Slip[PoPn_CatBoost.argmax():]
    PoPn_CatBoost_des=PoPn_CatBoost[PoPn_CatBoost.argmax():]
else: 
    Slip_CatBoost_des=np.array([np.nan]) 
    PoPn_CatBoost_des=np.array([np.nan])  

Slip_NGBoost_asc=Slip[0:PoPn_NGBoost.argmax()+1]
PoPn_NGBoost_asc=PoPn_NGBoost[0:PoPn_NGBoost.argmax()+1]
if PoPn_NGBoost.argmax()<PoPn_NGBoost.shape[0]-1:
    Slip_NGBoost_des=Slip[PoPn_NGBoost.argmax():]
    PoPn_NGBoost_des=PoPn_NGBoost[PoPn_NGBoost.argmax():]
else: 
    Slip_NGBoost_des=np.array([np.nan]) 
    PoPn_NGBoost_des=np.array([np.nan])  

if AISC_AL=='Yes' and EC4_AL=='Yes': Pu_options=['EC4 model','AISC 360 model','DH model','User-specified']
elif AISC_AL=='No' and EC4_AL=='Yes': Pu_options=['EC4 model','DH model','User-specified']
elif AISC_AL=='Yes' and EC4_AL=='No': Pu_options=['AISC 360 model','DH model','User-specified']
elif AISC_AL=='No' and EC4_AL=='No': Pu_options=['DH model','User-specified']

if EC4_AL=='Yes': PRk_options=['EC4 model','DH model','Modified DH model','User-specified']
elif EC4_AL=='No': PRk_options=['DH model','Modified DH model','User-specified']

du_ksc_options=['Based on relative load','Based on characteristic resistance']


du_ksc_selection=st.sidebar.selectbox('Select the method for determining the slip capacity and shear stiffness', options=du_ksc_options)
if du_ksc_selection=='Based on relative load': 
    PoPu_du=st.sidebar.number_input('Specify $P/P_\mathrm{u}$ at which $\delta_\mathrm{u}$ to be determined', min_value=0.0, max_value=1.0, value=0.95)
    PoPu_ksc_min=st.sidebar.number_input('Specify the minimum $P/P_\mathrm{u}$ value of the range at which $k_\mathrm{sc}$ to be determined', min_value=0.0, max_value=0.99, value=0.10)
    PoPu_ksc_max=st.sidebar.number_input('Specify the maximum $P/P_\mathrm{u}$ value of the range at which $k_\mathrm{sc}$ to be determined', min_value=min(0.0,PoPu_ksc_min+0.05), max_value=1.0, value=0.40)
    Pu_selection=st.sidebar.selectbox('Select the $P_\mathrm{u}$ determination method ', options=Pu_options)
    if units=='SI':
        if Pu_selection=='EC4 model': Pu=Pn_EC4
        elif Pu_selection=='AISC 360 model': Pu=Pn_AISC
        elif Pu_selection=='DH model': Pu=Pn_DH
        elif Pu_selection=='User-specified': Pu=st.sidebar.number_input('Specify $P_\mathrm{u}$ in kN',value=Pn_DH/1000)*1000
    else: 
        if Pu_selection=='EC4 model': 
            Pu_kips=Pn_EC4_kips
            Pu=Pu_kips*4.4482216*1000
        elif Pu_selection=='AISC 360 model': 
            Pu_kips=Pn_AISC_kips
            Pu=Pu_kips*4.4482216*1000
        elif Pu_selection=='DH model': 
            Pu_kips=Pn_DH_kips
            Pu=Pu_kips*4.4482216*1000
        elif Pu_selection=='User-specified': 
            Pu_kips=st.sidebar.number_input('Specify $P_\mathrm{u}$ in kips',value=Pn_DH_kips)
            Pu=Pu_kips*4.4482216*1000 

    if (PoPu_du<PoPn_CatBoost[-1]): du_CatBoost=Slip[-1].item()
    else: du_CatBoost=np.interp(PoPu_du,PoPn_CatBoost_des[::-1],Slip_CatBoost_des[::-1].reshape(-1,),right=np.nan,left=np.nan)

    if (PoPu_du<PoPn_NGBoost[-1]): du_NGBoost=Slip[-1].item()
    else: du_NGBoost=np.interp(PoPu_du,PoPn_NGBoost_des[::-1],Slip_NGBoost_des[::-1].reshape(-1,),right=np.nan,left=np.nan)
    
    se_CatBoost_min=np.interp(PoPu_ksc_min,PoPn_CatBoost_asc,Slip_CatBoost_asc.reshape(-1,),right=np.nan)
    se_CatBoost_max=np.interp(PoPu_ksc_max,PoPn_CatBoost_asc,Slip_CatBoost_asc.reshape(-1,),right=np.nan)
    ksc_CatBoost=(PoPu_ksc_max-PoPu_ksc_min)*Pu/(se_CatBoost_max-se_CatBoost_min)    
    
    se_NGBoost_min=np.interp(PoPu_ksc_min,PoPn_NGBoost_asc,Slip_NGBoost_asc.reshape(-1,),right=np.nan)
    se_NGBoost_max=np.interp(PoPu_ksc_max,PoPn_NGBoost_asc,Slip_NGBoost_asc.reshape(-1,),right=np.nan)
    ksc_NGBoost=(PoPu_ksc_max-PoPu_ksc_min)*Pu/(se_NGBoost_max-se_NGBoost_min)
   
    
elif du_ksc_selection=='Based on characteristic resistance':
    PRk_ratio=st.sidebar.number_input('Specify the load level reative to $P_\mathrm{Rk}$ at which $k_\mathrm{sc}$ to be determined', min_value=0.0, max_value=1.0, value=0.7)
    Pu_selection=st.sidebar.selectbox('Select the $P_\mathrm{u}$ determination method ', options=Pu_options)
    if units=='SI':
        if Pu_selection=='EC4 model': Pu=Pn_EC4
        elif Pu_selection=='AISC 360 model': Pu=Pn_AISC
        elif Pu_selection=='DH model': Pu=Pn_DH
        elif Pu_selection=='User-specified': Pu=st.sidebar.number_input('Specify $P_\mathrm{u}$ in kN',value=Pn_DH/1000)*1000
    else: 
        if Pu_selection=='EC4 model': 
            Pu_kips=Pn_EC4_kips
            Pu=Pu_kips*4.4482216*1000
        elif Pu_selection=='AISC 360 model': 
            Pu_kips=Pn_AISC_kips
            Pu=Pu_kips*4.4482216*1000
        elif Pu_selection=='DH model': 
            Pu_kips=Pn_DH_kips
            Pu=Pu_kips*4.4482216*1000
        elif Pu_selection=='User-specified': 
            Pu_kips=st.sidebar.number_input('Specify $P_\mathrm{u}$ in kips',value=Pn_DH_kips)
            Pu=Pu_kips*4.4482216*1000            
    PRk_selection=st.sidebar.selectbox('Select the $P_\mathrm{Rk}$ determination method ', options=PRk_options)
    if units=='SI':
        if PRk_selection=='EC4 model': PRk=PRk_EC4
        elif PRk_selection=='DH model': PRk=PRk_DH
        elif PRk_selection=='Modified DH model': PRk=PRk_DH_mod        
        elif PRk_selection=='User-specified': PRk=st.sidebar.number_input('Specify $P_\mathrm{Rk}$ in kN',min_value=0.0, max_value=Pu/1000, value=PRk_DH/1000)*1000
    else: 
        if PRk_selection=='EC4 model': 
            PRk_kips=PRk_EC4_kips
            PRk=PRk_kips*4.4482216*1000
        elif PRk_selection=='DH model': 
            PRk_kips=PRk_DH_kips
            PRk=PRk_kips*4.4482216*1000
        elif PRk_selection=='Modified DH model': 
            PRk_kips=PRk_DH_mod_kips
            PRk=PRk_kips*4.4482216*1000            
        elif PRk_selection=='User-specified': 
            PRk_kips=st.sidebar.number_input('Specify $P_\mathrm{Rk}$ in kips',min_value=0.0, max_value=Pu_kips, value=PRk_DH_kips)
            PRk=PRk_kips*4.4482216*1000            

    if (PRk/Pu<PoPn_CatBoost[-1]): du_CatBoost=Slip[-1].item()
    else: du_CatBoost=np.interp(PRk/Pu,PoPn_CatBoost_des[::-1],Slip_CatBoost_des[::-1].reshape(-1,),right=np.nan,left=np.nan)
 
    se_CatBoost=np.interp(PRk_ratio*PRk/Pu,PoPn_CatBoost_asc,Slip_CatBoost_asc.reshape(-1,),right=np.nan)
    ksc_CatBoost=PRk_ratio*PRk/se_CatBoost
    
    if (PRk/Pu<PoPn_NGBoost[-1]): du_NGBoost=Slip[-1].item()
    else: du_NGBoost=np.interp(PRk/Pu,PoPn_NGBoost_des[::-1],Slip_NGBoost_des[::-1].reshape(-1,),right=np.nan,left=np.nan)
 
    se_NGBoost=np.interp(PRk_ratio*PRk/Pu,PoPn_NGBoost_asc,Slip_NGBoost_asc.reshape(-1,),right=np.nan)
    ksc_NGBoost=PRk_ratio*PRk/se_NGBoost
 
if units=='SI':

    st.write('##### Specified properties affecting the reative load-slip ($P/P_\mathrm{u}-\delta$) curve')
 
    st.write('Stud shank diameter: $d$=', "{:.0f}".format(d),' mm')
    st.write('Stud height: $h_\mathrm{sc}$=', "{:.0f}".format(hsc),' mm')
    st.write('Concrete compressive strength: $f_\mathrm{cm}$=', "{:.0f}".format(fcm),' MPa') 
    st.write('Longitudinal distance from the rib top corner to the nearest stud center: $e_\mathrm{t}$=', "{:.2f}".format(et),' mm')
    st.write('Transverse stud spacing in a rib: $s_\mathrm{y}$=', "{:.2f}".format(sy),' mm')    
    st.write('Normal force on the face of the slab as a percentage of longitudinal force: $N_\mathrm{norm}$=', "{:.2f}".format(NnormP),' %') 

    st.write('##### Other properties specified to compute the shear stiffness and slip capacity')

    st.write('Deck height: $h_\mathrm{pn}$=', "{:.2f}".format(hpn),' mm')
    st.write('Deck thickness: $t$=', "{:.2f}".format(t),' mm')
    st.write('Concrete rib width measured at the deck bottom: $b_\mathrm{bot}$=', "{:.2f}".format(bbot),' mm')
    st.write('Concrete rib width measured at the deck top: $b_\mathrm{top}$=', "{:.2f}".format(btop),' mm')  
    st.write('Concrete rib width measured at the deck mid-height: $b_\mathrm{0}$=', "{:.2f}".format(b0),' mm')        
    st.write('Number of studs in a rib: $n_\mathrm{r}$=', "{:.0f}".format(nr))
    st.write('Stud tensile strength: $f_\mathrm{u}$=', "{:.0f}".format(fu),' MPa')
    st.write('Stud height-to-diameter ratio: $h_\mathrm{sc}/d$=', "{:.2f}".format(hsc/d))
    st.write('Stud embedment depth-to-diameter ratio: $h_\mathrm{A}/d$=', "{:.2f}".format((hsc-hpn)/d))
    st.write('Stud welding type: $SW$=', SW)
    st.write('Longitudinal stud spacing in a rib: $s_\mathrm{y}$=', "{:.2f}".format(sx),' mm') 
    st.write('Stud position: $SP$=', Stud_Pos) 

    if EC4_AL=='Yes': st.write('The specified parameters are within the EC4 applicability limits')
    elif EC4_AL=='No': st.write('The specified parameters are outside the EC4 applicability limits')    
    
    if AISC_AL=='Yes': st.write('The specified parameters are within the AISC 360 applicability limits')
    elif AISC_AL=='No': st.write('The specified parameters are outside the AISC 360 applicability limits')        
    
    st.write('##### Predicted load-slip curves')
    
    df_PoPn_CatBoost_data=np.concatenate((Slip.reshape(-1,1),PoPn_CatBoost.reshape(-1,1)),axis=1)
    df_PoPn_CatBoost=pd.DataFrame(data=df_PoPn_CatBoost_data,columns=['Slip (mm)','P/Pu CatBoost'])
    df_PoPn_CatBoost_as_csv=df_PoPn_CatBoost.to_csv(index=False)    
    
    df_PoPn_NGBoost_data=np.concatenate((Slip.reshape(-1,1),PoPn_NGBoost.reshape(-1,1),Predictions_68p3lower.reshape(-1,1),Predictions_95p4lower.reshape(-1,1),Predictions_99p7lower.reshape(-1,1),Predictions_68p3upper.reshape(-1,1),Predictions_95p4upper.reshape(-1,1),Predictions_99p7upper.reshape(-1,1)),axis=1)
    df_PoPn_NGBoost=pd.DataFrame(data=df_PoPn_NGBoost_data,columns=['Slip (mm)','P/Pu NGBoost, mean','P/Pu NGBoost, 68.3% confidence interval, lower','P/Pu NGBoost, 95.4% confidence interval, lower','P/Pu NGBoost, 99.7% confidence interval, lower','P/Pu NGBoost, 68.3% confidence interval, upper','P/Pu NGBoost, 95.4% confidence interval, upper','P/Pu NGBoost, 99.7% confidence interval, upper'])
    df_PoPn_NGBoost_as_csv=df_PoPn_NGBoost.to_csv(index=False)

    f1 = plt.figure(figsize=(3.74*2,2*2.25), dpi=200)
    
    ax1 = f1.add_subplot(2,2,1)
    ax1.plot(Slip, PoPn_CatBoost, color='b',linewidth=0.75, linestyle='solid',label='CatBoost')
    ax1.set_xlabel('$\delta$ (mm)', fontsize=10)
    ax1.set_ylabel('$P/P_\mathrm{u}$', fontsize=10)
    ax1.set_xlim([0, (Slip[-1]+0.25)])
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='both', which='major', labelsize=8, direction='in', width=0.5,length=2)
    for axis in ['top','bottom','left','right']:
      ax1.spines[axis].set_linewidth(0.5)
    ax1.grid(color='grey', linestyle='dotted', linewidth=0.3)
    ax1.legend(fontsize=8,ncol=1).get_frame().set_linewidth(0.25)
    
    ax2 = f1.add_subplot(2,2,2)
    ax2.plot(Slip, PoPn_CatBoost*Pu/1000, color='b',linewidth=0.75, linestyle='solid',label='CatBoost')
    if du_ksc_selection=='Based on relative load':
        ax2.hlines(PoPu_du*Pu/(1000),xmin=0,xmax=Slip[-1],colors='r',linewidth=0.75, linestyle='dotted')
    elif du_ksc_selection=='Based on characteristic resistance':
        ax2.hlines(PRk/(1000),xmin=0,xmax=Slip[-1],colors='r',linewidth=0.75, linestyle='dotted')
        
    if du_ksc_selection=='Based on relative load':
        ax2.plot([0,(PoPu_du*Pu/ksc_CatBoost)],[0,PoPu_du*Pu/(1000)],color='r',linewidth=0.75, linestyle='dotted')
    elif du_ksc_selection=='Based on characteristic resistance':
        ax2.plot([0,(PRk/ksc_CatBoost)],[0,PRk/(1000)],color='r',linewidth=0.75, linestyle='dotted') 
    ax2.set_xlabel('$\delta$ (mm)', fontsize=10)
    ax2.set_ylabel('$P$ (kN)', fontsize=10)
    ax2.set_xlim([0, (Slip[-1]+0.25)])
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='both', which='major', labelsize=8, direction='in', width=0.5,length=2)
    for axis in ['top','bottom','left','right']:
      ax2.spines[axis].set_linewidth(0.5)
    ax2.grid(color='grey', linestyle='dotted', linewidth=0.3)
    ax2.legend(fontsize=8,ncol=1).get_frame().set_linewidth(0.25)
    
    ax3 = f1.add_subplot(2,2,3)
    ax3.plot(Slip, PoPn_NGBoost, color='b',linewidth=0.75, linestyle='solid',label='NGBoost')

    ax3.fill_between(Slip.reshape(-1,), Predictions_68p3lower.reshape(-1,), Predictions_68p3upper.reshape(-1,),color= "#E49EDD",alpha= 0.8,label="68.3% confidence interval", lw=0.25)
    ax3.fill_between(Slip.reshape(-1,), Predictions_68p3lower.reshape(-1,), Predictions_95p4lower.reshape(-1,),color= "#E49EDD",alpha= 0.5,label="95.4% confidence interval", lw=0.25)
    ax3.fill_between(Slip.reshape(-1,), Predictions_68p3upper.reshape(-1,), Predictions_95p4upper.reshape(-1,),color= "#E49EDD",alpha= 0.5, lw=0.25)
    ax3.fill_between(Slip.reshape(-1,), Predictions_95p4lower.reshape(-1,), Predictions_99p7lower.reshape(-1,),color= "#E49EDD",alpha= 0.2,label="99.7% confidence interval", lw=0.25)
    ax3.fill_between(Slip.reshape(-1,), Predictions_95p4upper.reshape(-1,), Predictions_99p7upper.reshape(-1,),color= "#E49EDD",alpha= 0.2, lw=0.25)
    ax3.set_xlabel('$\delta$ (mm)', fontsize=10)
    ax3.set_ylabel('$P/P_\mathrm{u}$', fontsize=10)
    ax3.set_xlim([0, (Slip[-1]+0.25)])
    ax3.set_ylim(bottom=0)
    ax3.tick_params(axis='both', which='major', labelsize=8, direction='in', width=0.5,length=2)
    for axis in ['top','bottom','left','right']:
      ax3.spines[axis].set_linewidth(0.5)
    ax3.grid(color='grey', linestyle='dotted', linewidth=0.3)
    ax3.legend(fontsize=8,ncol=1).get_frame().set_linewidth(0.25) 

    ax4 = f1.add_subplot(2,2,4)
    ax4.plot(Slip, PoPn_NGBoost*Pu/1000, color='b',linewidth=0.75, linestyle='solid',label='NGBoost')
    if du_ksc_selection=='Based on relative load':
        ax4.hlines(PoPu_du*Pu/(1000),xmin=0,xmax=Slip[-1],colors='r',linewidth=0.75, linestyle='dotted')
    elif du_ksc_selection=='Based on characteristic resistance':
        ax4.hlines(PRk/(1000),xmin=0,xmax=Slip[-1],colors='r',linewidth=0.75, linestyle='dotted')
        
    if du_ksc_selection=='Based on relative load':
        ax4.plot([0,(PoPu_du*Pu/ksc_NGBoost)],[0,PoPu_du*Pu/(1000)],color='r',linewidth=0.75, linestyle='dotted')
    elif du_ksc_selection=='Based on characteristic resistance':
        ax4.plot([0,(PRk/ksc_NGBoost)],[0,PRk/(1000)],color='r',linewidth=0.75, linestyle='dotted') 
    ax4.fill_between(Slip.reshape(-1,), Predictions_68p3lower.reshape(-1,)*Pu/1000, Predictions_68p3upper.reshape(-1,)*Pu/1000,color= "#E49EDD",alpha= 0.8,label="68.3% confidence interval", lw=0.25)
    ax4.fill_between(Slip.reshape(-1,), Predictions_68p3lower.reshape(-1,)*Pu/1000, Predictions_95p4lower.reshape(-1,)*Pu/1000,color= "#E49EDD",alpha= 0.5,label="95.4% confidence interval", lw=0.25)
    ax4.fill_between(Slip.reshape(-1,), Predictions_68p3upper.reshape(-1,)*Pu/1000, Predictions_95p4upper.reshape(-1,)*Pu/1000,color= "#E49EDD",alpha= 0.5, lw=0.25)
    ax4.fill_between(Slip.reshape(-1,), Predictions_95p4lower.reshape(-1,)*Pu/1000, Predictions_99p7lower.reshape(-1,)*Pu/1000,color= "#E49EDD",alpha= 0.2,label="99.7% confidence interval", lw=0.25)
    ax4.fill_between(Slip.reshape(-1,), Predictions_95p4upper.reshape(-1,)*Pu/1000, Predictions_99p7upper.reshape(-1,)*Pu/1000,color= "#E49EDD",alpha= 0.2, lw=0.25)
    ax4.set_xlabel('$\delta$ (mm)', fontsize=10)
    ax4.set_ylabel('$P$ (kN)', fontsize=10)
    ax4.set_xlim([0, (Slip[-1]+0.25)])
    ax4.set_ylim(bottom=0)
    ax4.tick_params(axis='both', which='major', labelsize=8, direction='in', width=0.5,length=2)
    for axis in ['top','bottom','left','right']:
      ax4.spines[axis].set_linewidth(0.5)
    ax4.grid(color='grey', linestyle='dotted', linewidth=0.3)
    ax4.legend(fontsize=8,ncol=1).get_frame().set_linewidth(0.25)   

    f1.tight_layout()
    st.pyplot(f1)
    
    st.download_button(
    label="Download CatBoost P/Pu data as CSV",
    data=df_PoPn_CatBoost_as_csv,
    file_name="LSC_CatBoost_PoPu.csv",
    mime="text/csv",)    
    
    st.download_button(
    label="Download NGBoost P/Pu data as CSV",
    data=df_PoPn_NGBoost_as_csv,
    file_name="LSC_NGBoost_PoPu.csv",
    mime="text/csv",)  
   
    st.write('##### Slip capacity and shear stiffness based on the predicted load-slip curves')
    
    if du_ksc_selection=='Based on relative load':
        st.write('Slip capacity (CatBoost), $\delta_\mathrm{u}$=', "{:.1f}".format(du_CatBoost),' mm at $P/P_\mathrm{u}$=',"{:.2f}".format(PoPu_du))        
        st.write('Slip capacity (NGBoost), $\delta_\mathrm{u}$=', "{:.1f}".format(du_NGBoost),' mm at $P/P_\mathrm{u}$=',"{:.2f}".format(PoPu_du))
        st.write('Shear stiffness (CatBoost), $k_\mathrm{sc}$=', "{:.1f}".format(0.001*ksc_CatBoost),' kN/mm within the $P/P_\mathrm{u}$ range from ',"{:.2f}".format(PoPu_ksc_min),' to ',"{:.2f}".format(PoPu_ksc_max), 'with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/1000),' kN')         
        st.write('Shear stiffness (NGBoost), $k_\mathrm{sc}$=', "{:.1f}".format(0.001*ksc_NGBoost),' kN/mm within the $P/P_\mathrm{u}$ range from ',"{:.2f}".format(PoPu_ksc_min),' to ',"{:.2f}".format(PoPu_ksc_max), 'with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/1000),' kN') 
    elif du_ksc_selection=='Based on characteristic resistance':
        st.write('Slip capacity (CatBoost), $\delta_\mathrm{u}$=', "{:.1f}".format(du_CatBoost),' mm at $P_\mathrm{Rk}$=',"{:.2f}".format(PRk/1000),' kN, with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/1000),' kN') 
        st.write('Slip capacity (NGBoost), $\delta_\mathrm{u}$=', "{:.1f}".format(du_NGBoost),' mm at $P_\mathrm{Rk}$=',"{:.2f}".format(PRk/1000),' kN, with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/1000),' kN') 
        st.write('Shear stiffness (CatBoost), $k_\mathrm{sc}$=', "{:.1f}".format(0.001*ksc_CatBoost),' kN/mm at ',"{:.2f}".format(PRk_ratio),'$P_\mathrm{Rk}$=',"{:.2f}".format(PRk_ratio*PRk/1000),' kN, with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/1000),' kN')         
        st.write('Shear stiffness (NGBoost), $k_\mathrm{sc}$=', "{:.1f}".format(0.001*ksc_NGBoost),' kN/mm at ',"{:.2f}".format(PRk_ratio),'$P_\mathrm{Rk}$=',"{:.2f}".format(PRk_ratio*PRk/1000),' kN, with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/1000),' kN') 
      
    
else:
    st.write('##### Specified properties affecting the reative load-slip ($P/P_\mathrm{u}-\delta$) curve')
 
    st.write('Stud shank diameter: $d$=', "{:.3f}".format(d_in),' in.')
    st.write('Stud height: $h_\mathrm{sc}$=', "{:.3f}".format(hsc_in),' in.')
    st.write('Concrete compressive strength: $f^\prime_\mathrm{c}$=', "{:.0f}".format(fcm_psi),' psi') 
    st.write('Longitudinal distance from the rib top corner to the nearest stud center: $e_\mathrm{t}$=', "{:.3f}".format(et_in),' in.')
    st.write('Transverse stud spacing in a rib: $s_\mathrm{y}$=', "{:.3f}".format(sy_in),' in.')    
    st.write('Normal force on the face of the slab as a percentage of longitudinal force: $N_\mathrm{norm}$=', "{:.2f}".format(NnormP),' %') 
    
    st.write('##### Other properties specified to compute the shear stiffness and slip capacity')

    st.write('Deck height: $h_\mathrm{pn}$=', "{:.3f}".format(hpn_in),' in.')
    st.write('Deck thickness: $t$=', "{:.4f}".format(t_in),' in.')
    st.write('Concrete rib width measured at the deck bottom: $b_\mathrm{bot}$=', "{:.3f}".format(bbot_in),' in.')
    st.write('Concrete rib width measured at the deck top: $b_\mathrm{top}$=', "{:.3f}".format(btop_in),' in.')  
    st.write('Concrete rib width measured at the deck mid-height: $b_\mathrm{0}$=', "{:.3f}".format(b0_in),' in.')        
    st.write('Number of studs in a rib: $n_\mathrm{r}$=', "{:.0f}".format(nr))
    st.write('Stud tensile strength: $f_\mathrm{u}$=', "{:.3f}".format(fu_ksi),' ksi')
    st.write('Stud height-to-diameter ratio: $h_\mathrm{sc}/d$=', "{:.2f}".format(hsc/d))
    st.write('Stud embedment depth-to-diameter ratio: $h_\mathrm{A}/d$=', "{:.2f}".format((hsc-hpn)/d))
    st.write('Stud welding type: $SW$=', SW)
    st.write('Longitudinal stud spacing in a rib: $s_\mathrm{y}$=', "{:.3f}".format(sx_in),' in.') 
    st.write('Stud position: $SP$=', Stud_Pos) 

    if EC4_AL=='Yes': st.write('The specified parameters are within the EC4 applicability limits')
    elif EC4_AL=='No': st.write('The specified parameters are outside the EC4 applicability limits')    
    
    if AISC_AL=='Yes': st.write('The specified parameters are within the AISC 360 applicability limits')
    elif AISC_AL=='No': st.write('The specified parameters are outside the AISC 360 applicability limits')  
    
    st.write('##### Predicted load-slip curve')
    
    df_PoPn_CatBoost_data=np.concatenate((Slip.reshape(-1,1)/25.4,PoPn_CatBoost.reshape(-1,1)),axis=1)
    df_PoPn_CatBoost=pd.DataFrame(data=df_PoPn_CatBoost_data,columns=['Slip (in.)','P/Pu CatBoost'])
    df_PoPn_CatBoost_as_csv=df_PoPn_CatBoost.to_csv(index=False)    
    
    df_PoPn_NGBoost_data=np.concatenate((Slip.reshape(-1,1)/25.4,PoPn_NGBoost.reshape(-1,1),Predictions_68p3lower.reshape(-1,1),Predictions_95p4lower.reshape(-1,1),Predictions_99p7lower.reshape(-1,1),Predictions_68p3upper.reshape(-1,1),Predictions_95p4upper.reshape(-1,1),Predictions_99p7upper.reshape(-1,1)),axis=1)
    df_PoPn_NGBoost=pd.DataFrame(data=df_PoPn_NGBoost_data,columns=['Slip (in.)','P/Pu NGBoost, mean','P/Pu NGBoost, 68.3% confidence interval, lower','P/Pu NGBoost, 95.4% confidence interval, lower','P/Pu NGBoost, 99.7% confidence interval, lower','P/Pu NGBoost, 68.3% confidence interval, upper','P/Pu NGBoost, 95.4% confidence interval, upper','P/Pu NGBoost, 99.7% confidence interval, upper'])
    df_PoPn_NGBoost_as_csv=df_PoPn_NGBoost.to_csv(index=False)
   
    f1 = plt.figure(figsize=(3.74*2,2*2.25), dpi=200)
    
    ax1 = f1.add_subplot(2,2,1)
    ax1.plot(Slip/25.4, PoPn_CatBoost, color='b',linewidth=0.75, linestyle='solid',label='CatBoost')
 
    ax1.set_xlabel('$\delta$ (in.)', fontsize=10)
    ax1.set_ylabel('$P/P_\mathrm{u}$', fontsize=10)
    ax1.set_xlim([0, (Slip[-1]+0.25)/25.4])
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='both', which='major', labelsize=8, direction='in', width=0.5,length=2)
    for axis in ['top','bottom','left','right']:
      ax1.spines[axis].set_linewidth(0.5)
    ax1.grid(color='grey', linestyle='dotted', linewidth=0.3)
    ax1.legend(fontsize=8,ncol=1).get_frame().set_linewidth(0.25)
    
    ax2 = f1.add_subplot(2,2,2)
    ax2.plot(Slip/25.4, PoPn_CatBoost*Pu_kips, color='b',linewidth=0.75, linestyle='solid',label='CatBoost')
    if du_ksc_selection=='Based on relative load':
        ax2.hlines(PoPu_du*Pu/(4.4482216*1000),xmin=0,xmax=Slip[-1]/25.4,colors='r',linewidth=0.75, linestyle='dotted')
    elif du_ksc_selection=='Based on characteristic resistance':
        ax2.hlines(PRk/(4.4482216*1000),xmin=0,xmax=Slip[-1]/25.4,colors='r',linewidth=0.75, linestyle='dotted')
        
    if du_ksc_selection=='Based on relative load':
        ax2.plot([0,(PoPu_du*Pu/ksc_CatBoost)/25.4],[0,PoPu_du*Pu/(4.4482216*1000)],color='r',linewidth=0.75, linestyle='dotted')
    elif du_ksc_selection=='Based on characteristic resistance':
        ax2.plot([0,(PRk/ksc_CatBoost)/25.4],[0,PRk/(4.4482216*1000)],color='r',linewidth=0.75, linestyle='dotted')        

    ax2.set_xlabel('$\delta$ (in.)', fontsize=10)
    ax2.set_ylabel('$P$ (kips)', fontsize=10)
    ax2.set_xlim([0, (Slip[-1]+0.25)/25.4])
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='both', which='major', labelsize=8, direction='in', width=0.5,length=2)
    for axis in ['top','bottom','left','right']:
      ax2.spines[axis].set_linewidth(0.5)
    ax2.grid(color='grey', linestyle='dotted', linewidth=0.3)
    ax2.legend(fontsize=8,ncol=1).get_frame().set_linewidth(0.25)   

    ax3 = f1.add_subplot(2,2,3)
    ax3.plot(Slip/25.4, PoPn_NGBoost, color='b',linewidth=0.75, linestyle='solid',label='NGBoost')
 
    ax3.fill_between(Slip.reshape(-1,)/25.4, Predictions_68p3lower.reshape(-1,), Predictions_68p3upper.reshape(-1,),color= "#E49EDD",alpha= 0.8,label="68.3% confidence interval", lw=0.25)
    ax3.fill_between(Slip.reshape(-1,)/25.4, Predictions_68p3lower.reshape(-1,), Predictions_95p4lower.reshape(-1,),color= "#E49EDD",alpha= 0.5,label="95.4% confidence interval", lw=0.25)
    ax3.fill_between(Slip.reshape(-1,)/25.4, Predictions_68p3upper.reshape(-1,), Predictions_95p4upper.reshape(-1,),color= "#E49EDD",alpha= 0.5, lw=0.25)
    ax3.fill_between(Slip.reshape(-1,)/25.4, Predictions_95p4lower.reshape(-1,), Predictions_99p7lower.reshape(-1,),color= "#E49EDD",alpha= 0.2,label="99.7% confidence interval", lw=0.25)
    ax3.fill_between(Slip.reshape(-1,)/25.4, Predictions_95p4upper.reshape(-1,), Predictions_99p7upper.reshape(-1,),color= "#E49EDD",alpha= 0.2, lw=0.25)
    ax3.set_xlabel('$\delta$ (in.)', fontsize=10)
    ax3.set_ylabel('$P/P_\mathrm{u}$', fontsize=10)
    ax3.set_xlim([0, (Slip[-1]+0.25)/25.4])
    ax3.set_ylim(bottom=0)
    ax3.tick_params(axis='both', which='major', labelsize=8, direction='in', width=0.5,length=2)
    for axis in ['top','bottom','left','right']:
      ax3.spines[axis].set_linewidth(0.5)
    ax3.grid(color='grey', linestyle='dotted', linewidth=0.3)
    ax3.legend(fontsize=8,ncol=1).get_frame().set_linewidth(0.25)  

    ax4 = f1.add_subplot(2,2,4)
    ax4.plot(Slip/25.4, PoPn_NGBoost*Pu_kips, color='b',linewidth=0.75, linestyle='solid',label='NGBoost')
    if du_ksc_selection=='Based on relative load':
        ax4.hlines(PoPu_du*Pu/(4.4482216*1000),xmin=0,xmax=Slip[-1]/25.4,colors='r',linewidth=0.75, linestyle='dotted')
    elif du_ksc_selection=='Based on characteristic resistance':
        ax4.hlines(PRk/(4.4482216*1000),xmin=0,xmax=Slip[-1]/25.4,colors='r',linewidth=0.75, linestyle='dotted')
        
    if du_ksc_selection=='Based on relative load':
        ax4.plot([0,(PoPu_du*Pu/ksc_NGBoost)/25.4],[0,PoPu_du*Pu/(4.4482216*1000)],color='r',linewidth=0.75, linestyle='dotted')
    elif du_ksc_selection=='Based on characteristic resistance':
        ax4.plot([0,(PRk/ksc_NGBoost)/25.4],[0,PRk/(4.4482216*1000)],color='r',linewidth=0.75, linestyle='dotted')        

    ax4.fill_between(Slip.reshape(-1,)/25.4, Predictions_68p3lower.reshape(-1,)*Pu_kips, Predictions_68p3upper.reshape(-1,)*Pu_kips,color= "#E49EDD",alpha= 0.8,label="68.3% confidence interval", lw=0.25)
    ax4.fill_between(Slip.reshape(-1,)/25.4, Predictions_68p3lower.reshape(-1,)*Pu_kips, Predictions_95p4lower.reshape(-1,)*Pu_kips,color= "#E49EDD",alpha= 0.5,label="95.4% confidence interval", lw=0.25)
    ax4.fill_between(Slip.reshape(-1,)/25.4, Predictions_68p3upper.reshape(-1,)*Pu_kips, Predictions_95p4upper.reshape(-1,)*Pu_kips,color= "#E49EDD",alpha= 0.5, lw=0.25)
    ax4.fill_between(Slip.reshape(-1,)/25.4, Predictions_95p4lower.reshape(-1,)*Pu_kips, Predictions_99p7lower.reshape(-1,)*Pu_kips,color= "#E49EDD",alpha= 0.2,label="99.7% confidence interval", lw=0.25)
    ax4.fill_between(Slip.reshape(-1,)/25.4, Predictions_95p4upper.reshape(-1,)*Pu_kips, Predictions_99p7upper.reshape(-1,)*Pu_kips,color= "#E49EDD",alpha= 0.2, lw=0.25)
    ax4.set_xlabel('$\delta$ (in.)', fontsize=10)
    ax4.set_ylabel('$P$ (kips)', fontsize=10)
    ax4.set_xlim([0, (Slip[-1]+0.25)/25.4])
    ax4.set_ylim(bottom=0)
    ax4.tick_params(axis='both', which='major', labelsize=8, direction='in', width=0.5,length=2)
    for axis in ['top','bottom','left','right']:
      ax4.spines[axis].set_linewidth(0.5)
    ax4.grid(color='grey', linestyle='dotted', linewidth=0.3)
    ax4.legend(fontsize=8,ncol=1).get_frame().set_linewidth(0.25)       
    
    f1.tight_layout()
    st.pyplot(f1)
    
    st.download_button(
    label="Download CatBoost P/Pu data as CSV",
    data=df_PoPn_CatBoost_as_csv,
    file_name="LSC_CatBoost_PoPu.csv",
    mime="text/csv",)      
    
    st.download_button(
    label="Download NGBoost P/Pu data as CSV",
    data=df_PoPn_NGBoost_as_csv,
    file_name="LSC_NGBoost_PoPu.csv",
    mime="text/csv",)    
   
    st.write('##### Slip capacity and shear stiffness based on the predicted load-slip curves')
     
    if du_ksc_selection=='Based on relative load':
        st.write('Slip capacity (CatBoost), $\delta_\mathrm{u}$=', "{:.3f}".format(du_CatBoost/25.4),' in. at $P/P_\mathrm{u}$=',"{:.2f}".format(PoPu_du))        
        st.write('Slip capacity (NGBoost), $\delta_\mathrm{u}$=', "{:.3f}".format(du_NGBoost/25.4),' in. at $P/P_\mathrm{u}$=',"{:.2f}".format(PoPu_du))
        st.write('Shear stiffness (CatBoost), $k_\mathrm{sc}$=', "{:.1f}".format(0.001*ksc_CatBoost*5.71),' kip/in. within the $P/P_\mathrm{u}$ range from ',"{:.2f}".format(PoPu_ksc_min),' to ',"{:.2f}".format(PoPu_ksc_max), 'with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/(4.4482216*1000)),' kips')         
        st.write('Shear stiffness (NGBoost), $k_\mathrm{sc}$=', "{:.1f}".format(0.001*ksc_NGBoost*5.71),' kip/in. within the $P/P_\mathrm{u}$ range from ',"{:.2f}".format(PoPu_ksc_min),' to ',"{:.2f}".format(PoPu_ksc_max), 'with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/(4.4482216*1000)),' kips') 
    elif du_ksc_selection=='Based on characteristic resistance':
        st.write('Slip capacity (CatBoost), $\delta_\mathrm{u}$=', "{:.3f}".format(du_CatBoost/25.4),' in. at $P_\mathrm{Rk}$=',"{:.2f}".format(PRk/(4.4482216*1000)),' kips, with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/(4.4482216*1000)),' kips')            
        st.write('Slip capacity (NGBoost), $\delta_\mathrm{u}$=', "{:.3f}".format(du_NGBoost/25.4),' in. at $P_\mathrm{Rk}$=',"{:.2f}".format(PRk/(4.4482216*1000)),' kips, with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/(4.4482216*1000)),' kips')    
        st.write('Shear stiffness (CatBoost), $k_\mathrm{sc}$=', "{:.1f}".format(0.001*ksc_CatBoost*5.71),' kip/in. at ',"{:.2f}".format(PRk_ratio),'$P_\mathrm{Rk}$=',"{:.2f}".format(PRk_ratio*PRk/(4.4482216*1000)),' kips, with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/(4.4482216*1000)),' kips') 

        st.write('Shear stiffness (NGBoost), $k_\mathrm{sc}$=', "{:.1f}".format(0.001*ksc_NGBoost*5.71),' kip/in. at ',"{:.2f}".format(PRk_ratio),'$P_\mathrm{Rk}$=',"{:.2f}".format(PRk_ratio*PRk/(4.4482216*1000)),' kips, with the assumed value of $P_\mathrm{u}$=',"{:.2f}".format(Pu/(4.4482216*1000)),' kips')             

