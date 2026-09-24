#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3 (V4.0) — Bayesian RI-CLPM posterior densities of the lagged within-person
engagement->DEP paths (gamma1: ln(Watch), gamma2: ln(Click)).

Data: ../data/figure3_posterior_density_digitized.csv
  The density curves were digitized from the V3.9 manuscript's embedded Figure 3
  render (which itself was produced from the actual NUTS posterior draws;
  pipeline: PyMC 5.28.4, tune=500, draws=1000, chains=2, seed=42).
  Anchor statistics verified against V3.9 Table 5:
    watch: gamma = -0.0004, 95% HDI [-0.0018, +0.0010], BF01 = 543.30
    click: gamma = -0.0002, 95% HDI [-0.0016, +0.0011], BF01 = 683.44
Output: 新Figure3_V4_proof.svg / .png (400 dpi)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAVY=(30/255,51/255,90/255); REDL=(165/255,32/255,38/255); GRAYT=(0.35,0.35,0.35)
BLUE=(31/255,119/255,180/255); RED3=(192/255,32/255,38/255)

panels={'gamma1_watch':[], 'gamma2_click':[]}
with open('../data/figure3_posterior_density_digitized.csv') as f:
    for row in csv.DictReader(f):
        panels[row['panel']].append((float(row['beta']),float(row['density'])))
panels={k:np.array(v).T for k,v in panels.items()}

fig,axs=plt.subplots(1,2,figsize=(13.2,4.6))
cfg=[(axs[0],panels['gamma1_watch'],BLUE,(0.68,0.82,0.92),"\u03b3\u2081: ln(Watch) \u2192 DEP",
      "BF\u2080\u2081 = 543.30",-0.0004,(-0.0018,0.0010),"95% HDI [\u22120.0018, +0.0010]"),
     (axs[1],panels['gamma2_click'],RED3,(0.96,0.80,0.80),"\u03b3\u2082: ln(Click) \u2192 DEP",
      "BF\u2080\u2081 = 683.44",-0.0002,(-0.0016,0.0011),"95% HDI [\u22120.0016, +0.0011]")]
for a,(x,y),c,fc,tit,bf,est,hdi,hdit in cfg:
    a.fill_between(x,y,color=fc,zorder=2)
    a.plot(x,y,color=c,lw=2.4,zorder=3,label='posterior density')
    a.axvline(est,color=(0.15,0.15,0.15),lw=1.8,ls=(0,(6,3)),zorder=4,label='posterior estimate')
    for h in hdi:
        a.axvline(h,color=c,lw=1.6,ls=(0,(2,2)),zorder=4)
    a.plot([],[],color=c,lw=1.6,ls=(0,(2,2)),label='95% HDI bounds')
    a.axvline(0,color=(0.45,0.45,0.45),lw=1.8,zorder=4,label='zero reference')
    a.set_title(tit,fontsize=13,fontweight='bold')
    a.set_xlim(-0.005,0.005); a.set_ylim(0,860); a.set_ylabel('Density',fontsize=11)
    a.tick_params(labelsize=9.5)
    a.text(0.03,0.94,bf,transform=a.transAxes,fontsize=12.5,fontweight='bold',
           style='italic',color=REDL,va='top')
    a.text(0.03,0.82,hdit,transform=a.transAxes,fontsize=10.5,color=(0.15,0.15,0.15),va='top')
    a.text(0.03,0.70,"Evidence favors H\u2080 under the stated\nmodel and prior",
           transform=a.transAxes,fontsize=9.8,style='italic',color=NAVY,va='top')
    a.legend(loc='upper right',fontsize=8.2,frameon=True,edgecolor=(0.7,0.7,0.7))
    for s in ['top','right']: a.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig('新Figure3_V4_proof.svg'); fig.savefig('新Figure3_V4_proof.png',dpi=400)
print('Figure 3 written.')
