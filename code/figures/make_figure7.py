#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7 (V4.0) — DADM schematic synthesis (conceptual figure, NOT estimated).

Curve data: ../data/figure7_schematic_curve_digitized.csv
  The blue curve shape was digitized from the approved V3.9-era schematic master
  and is explicitly labeled in-figure: "Schematic coordinates; curve shape and
  reference-line position are not estimated." It is a visual synthesis device,
  not a fitted dose-response function.
Empirical anchors shown (all verified against V3.9):
  BF01 = 543.30 (lagged WP), beta = +1.564 (extensive), beta = -0.492 (intensive),
  r = -0.202 (behavioral diversity).
Output: 新Figure7_V4_proof.svg / .png (400 dpi)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAVY=(30/255,51/255,90/255); REDL=(165/255,32/255,38/255)
GRAYT=(0.35,0.35,0.35); BLACK=(0.1,0.1,0.1); SH=(238/255,241/255,246/255)
GRN=(40/255,110/255,60/255); YELLB=(200/255,170/255,60/255)

uv=np.loadtxt('../data/figure7_schematic_curve_digitized.csv',delimiter=',',skiprows=1)
u,v=uv[:,0],uv[:,1]

fig,ax=plt.subplots(figsize=(16.8,13.16))
ax.set_xlim(0,1200); ax.set_ylim(0,940); ax.axis('off')

def tbox(x0,y0,x1,y1,ec,fc,lw=1.6,ls='-',r=6):
    ax.add_patch(FancyBboxPatch((x0,940-y1),x1-x0,y1-y0,
        boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,ls=ls,mutation_aspect=1))
def txt(x,y,s,fs,c=BLACK,b=False,i=False,ha='center'):
    ax.text(x,940-y,s,fontsize=fs,fontweight='bold' if b else 'normal',
            style='italic' if i else 'normal',color=c,ha=ha,va='center')

ax.add_patch(plt.Rectangle((45,940-890),1140,890-588,fc=SH,ec='none',zorder=1))
ax.annotate('',xy=(45,940-15),xytext=(45,940-905),arrowprops=dict(arrowstyle='-|>',color=BLACK,lw=1.6))
ax.annotate('',xy=(1198,940-890),xytext=(45,940-890),arrowprops=dict(arrowstyle='-|>',color=BLACK,lw=1.6))
ax.plot([45,1185],[940-285,940-285],color=(0.7,0.7,0.7),lw=1.2,ls=(0,(4,4)),zorder=2)
ax.plot(45+u*1140,(940-890)+v*(890-285),color=NAVY,lw=3.4,zorder=3,solid_capstyle='round')
ax.plot([130,1100],[940-890,940-150],color=(0.45,0.45,0.45),lw=2.6,ls=(0,(10,8)),zorder=3)
ax.plot([45,1185],[940-588,940-588],color=REDL,lw=3.2,zorder=4)
ax.annotate('',xy=(1106,940-592),xytext=(1106,940-648),
            arrowprops=dict(arrowstyle='-|>',color=REDL,lw=2.6),zorder=4)

ax.text(612,940-913,"Engagement Volume (observed activity intensity)",fontsize=15,fontweight='bold',color=BLACK,ha='center')
ax.text(612,940-933,"Schematic coordinates; curve shape and reference-line position are not estimated.",
        fontsize=9.5,style='italic',color=GRAYT,ha='center')
ax.text(24,470,"Behavioral Composition (comment share / DEP)",fontsize=14,fontweight='bold',
        color=BLACK,ha='center',va='center',rotation=90)

fl=["Schematic synthesis only; trajectories and reference lines are conceptual and",
    "are not empirically validated diagnostic thresholds. DEP is an observed",
    "behavioral-composition measure, not a direct measure of trust stage or",
    "relational depth."]
for j,s in enumerate(fl): ax.text(55,940-108-15*j,s,fontsize=8.6,color=GRAYT,ha='left',va='center')

lx,ly=375,112
ax.plot([lx,lx+55],[940-ly,940-ly],color=NAVY,lw=3.2)
ax.text(lx+65,940-ly,"schematic synthesis of observed results",fontsize=10.5,ha='left',va='center')
ax.plot([lx,lx+55],[940-ly-24,940-ly-24],color=(0.45,0.45,0.45),lw=2.4,ls=(0,(8,6)))
ax.text(lx+65,940-ly-24,"illustrative coupled-growth scenario",fontsize=10.5,ha='left',va='center')
ax.plot([lx,lx+55],[940-ly-48,940-ly-48],color=REDL,lw=3.2)
ax.text(lx+65,940-ly-48,"conceptual reference: composition held constant",fontsize=10.5,ha='left',va='center')

ax.text(883,940-32,"Behavioral-diversity association: higher engagement is associated with",fontsize=11.5,color=GRAYT,ha='center')
ax.text(883,940-50,"lower Shannon entropy (r = \u22120.202); algorithmic exposure is not observed.",fontsize=11.5,color=GRAYT,ha='center')

tbox(615,180,945,240,(0.4,0.4,0.4),(0.98,0.98,0.98),1.4)
txt(780,197,"Illustrative coupled-growth scenario \u2014",10.5,GRAYT,b=True)
txt(780,213,"conceptual reference only;",10.5,GRAYT)
txt(780,228,"not an estimated trajectory.",10.5,GRAYT)

tbox(370,435,590,520,YELLB,(252/255,246/255,220/255),1.8)
txt(480,452,"Mechanism remains open:",10,YELLB,b=True)
txt(480,467,"psychological processes underlying",9.2,YELLB)
txt(480,481,"volume\u2013composition divergence are",9.2,YELLB)
txt(480,495,"not directly observed.",9.2,YELLB)

txt(985,505,"CONCEPTUAL REFERENCE:",13,REDL,b=True)
txt(985,524,"COMPOSITION HELD CONSTANT",13,REDL,b=True)
txt(985,546,"not an estimated level or threshold",9.5,GRAYT,i=True)

tbox(455,716,735,760,GRN,(238/255,247/255,240/255),1.8)
txt(595,732,"Extensive margin: participation",10.5,GRN,b=True)
txt(595,749,"breadth increases (\u03b2 = +1.564)",10.5,GRN)

tbox(412,806,700,862,(0.4,0.4,0.4),(0.98,0.98,0.98),1.4)
txt(556,821,"Lagged within-person DEP association:",9.8,BLACK,b=True)
txt(556,836,"BF\u2080\u2081 = 543.30 \u2014 evidence favors the point-null",9.8,BLACK)
txt(556,850,"under the stated model and prior",9.8,BLACK)

tbox(874,652,1116,706,REDL,(253/255,240/255,240/255),1.8)
txt(995,666,"Intensive margin: conditional",9.8,REDL,b=True)
txt(995,680,"composition shifts downward",9.8,REDL)
txt(995,694,"(\u03b2 = \u22120.492)",9.8,REDL)

ax.text(55,940-862,"Higher engagement volume without",fontsize=11,fontweight='bold',color=NAVY,ha='left')
ax.text(55,940-880,"corresponding compositional increase",fontsize=11,fontweight='bold',color=NAVY,ha='left')

plt.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
fig.savefig('新Figure7_V4_proof.svg'); fig.savefig('新Figure7_V4_proof.png',dpi=400)
print('Figure 7 written.')
