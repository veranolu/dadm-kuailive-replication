#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 (V4.0) — Extensive-Intensive Margin Decomposition (forest plot).

Data: ../data/figure4_margin_estimates.csv
  beta = +1.564 (extensive, logit participation) and beta = -0.492
  (intensive, fractional conditional share), verified against V3.9 P123/P125.
  CI half-widths are the graphical intervals from the V3.9 embedded figure;
  exact SEs are reported in the manuscript tables (SM Table S19).
Output: 新Figure4_V4_proof.svg / .png (400 dpi)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REDL=(165/255,32/255,38/255); GRAYT=(0.35,0.35,0.35)
rows=[("Hurdle Stage 1 \u2014 Extensive margin\n(participation incidence, logit)",1.564,0.12,(0.13,0.35,0.65)),
      ("Hurdle Stage 2 \u2014 Intensive margin\n(conditional comment share | participants)",-0.492,0.06,REDL)]

fig,ax=plt.subplots(figsize=(12.8,6.0))
for i,(lab,b,ci,c) in enumerate(rows):
    y=1-i
    ax.errorbar(b,y,xerr=ci,fmt='D',color=c,ms=11,capsize=5,capthick=2,elinewidth=2.2,zorder=3)
    xa=b+ci+0.14 if b>0 else b-ci-0.14
    ax.text(xa,y+0.16,f"\u03b2 = {'+' if b>0 else '\u2212'}{abs(b):.3f}",fontsize=13,
            fontweight='bold',style='italic',color=c,ha='left' if b>0 else 'right')
ax.axvline(0,color=REDL,lw=1.6,ls=(0,(6,4)),zorder=2)
ax.set_yticks([1,0]); ax.set_yticklabels([r[0] for r in rows],fontsize=10.5)
ax.set_ylim(-0.55,1.55); ax.set_xlim(-1.0,2.0)
ax.set_xlabel("Engagement association estimate (\u03b2; component-specific scales)",fontsize=11.5)
ax.set_title("Extensive\u2013Intensive Margin Decomposition",fontsize=14,fontweight='bold')
ax.grid(axis='x',color=(0.9,0.9,0.9),lw=0.7,zorder=1)
for s in ['top','right','left']: ax.spines[s].set_visible(False)
fig.text(0.55,0.015,"Coefficients arise from different model components (logit participation vs. fractional conditional share)\n"
         "and should be interpreted by estimand and direction, not compared by absolute magnitude.",
         ha='center',fontsize=9.3,style='italic',color=GRAYT)
plt.tight_layout(rect=[0,0.09,1,1])
fig.savefig('新Figure4_V4_proof.svg'); fig.savefig('新Figure4_V4_proof.png',dpi=400)
print('Figure 4 written.')
