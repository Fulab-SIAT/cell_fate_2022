### nonlinear plasmid copy number model

We now consider the mechanisms for plasmid copy number controlling. For plasmid harboring the ColE1 origin, replication initiation process is controlled by RNA II-RNA I pairs. The copy number of plasmid $g_i$ satisfies the followed equation
$$
\frac{\mathrm{d} g_i}{\mathrm{d} t} = k_{II} Q_0 g_i - \lambda g_i
$$
where $k_{II}$ is the RNA II initiation rate, $Q_0$ is the DNA replication abolish term that describes the RNA I repression, and is given by
$$
Q_0 = \frac{1}{1 + R_{I} /K_{I}}
$$
We suppose the replication abolish process is a hyperbolic repression, $R_{I}$ is the concentration of RNA II, and $K_I$ is the inhibition constant. The concentration of RNA II, $R_{I}$ , can be described by followed equation:
$$
\frac{\mathrm{d} R_{I}}{\mathrm{d} t}  = (k_{I} - k_{II}) g_i - (\epsilon_{I} + \lambda) R_I
$$
We suppose the RNA II transcription process is faster than the plasmid replication process, and the degradation of RNA II is more rapidly than the growth dilution rate $\lambda$ , i.e. , $\epsilon_{I} >> \lambda $than Using the quasi-steady suppose, we have:
$$
R_{I}^{*} = g_i \cdot \Delta k/\epsilon_{I}
$$
Now, we have the simplified model
$$
\frac{\mathrm{d} g_{i}}{\mathrm{d} t} = k_{II} Q_{0}(g_{i}) g_{i} - \lambda g_{i}
$$
where, $g_{i}$ denotes the plasmid copy number, $R_p$ is the probability of the primer transcription results in replication. $\lambda$ is the growth rate. We rewrite the transcription rate of $k_{II}$ as $\alpha_{II}$.
$$
Q_{0}(g_i) = \frac{1}{1 + g_i / \widetilde{K_{I}}}
$$
where, $\widetilde{K_{I}} = K_{I} \epsilon_{I} / \Delta k$.

The equilibrium of the plasmid copy number is given by
$$
\frac{1}{\lambda} =  \frac{1}{\widetilde{K_{I}} k_{II}} g_{i}^{*} + \frac{1}{k_{II} }
$$
.





RNA II initials the plasmid replication
RNA I binds to RNA II forming complex.



<img src="nonlinear plasmid copy number model.assets/image-20211214100805225.png" alt="image-20211214100805225" style="zoom:15%;" />

  

The fitting $r^{2} = 0.93$.  the slope is 0.0187, and the interception is 0.283.

When the growth rate decreases, the slope becomes steep. It can be explained that the $k_{II}$ is the function of growth rate (RNA seq data of plasmid origin).

We suppose the $k_{II}$ is a constant, and $\rho = 1$. We have the $k_{II} = 3.54$ .

Thus, we have ${\Delta k}/({\epsilon_{I} K_{I}}) = 0.0663$ . we have the re-scaled $\widetilde{K_{I}} = 15.09$

Now, we have the simplified model
$$
\dot{g_{i}} = \alpha_{II} R_{p}(g_{i}) g_{i} - \lambda g_{i}
$$
where, $g_{i}$ denotes the plasmid copy number, $R_p$ is the probability of the primer transcription results in replication. $\lambda$ is the growth rate. We rewrite the transcription rate of $k_{II}$ as $\alpha_{II}$.
$$
R_{p}(g_i) = \frac{1}{1 + g_i / \widetilde{K_{I}}}
$$
<img src="nonlinear plasmid copy number model.assets/image-20211214095957639.png" alt="image-20211214095957639" style="zoom:15%;" />

 

> The code for simulating the plasmid copy number: sub_model/plasmid_control.py
>
> the code used in the Data Fitting Block



| Parameter            | Value | Unit     | Description |
| -------------------- | ----- | -------- | ----------- |
| $k_{II}$             | 3.54  | $h^{-1}$ |             |
| $\widetilde{K_{II}}$ | 15.09 | -        |             |
|                      |       |          |             |

### Why plasmid copy number that harboring ColE1 decreased in slow growth condition

Evidences:

1.  Škulj, M. *et al.* Improved determination of plasmid copy number using quantitative real-time PCR for monitoring fermentation processes. *Microb Cell Fact* **7**, 6 (2008).  
2.  Atlung, T., Christensen, B. B. & Hansen, F. G. Role of the Rom Protein in Copy Number Control of Plasmid pBR322 at Different Growth Rates inEscherichia coliK-12. *Plasmid* **41**, 110–119 (1999).  





### Notes

> 1. SNPs:
>
>    Sequencing reads were trimmed to remove Illumina specific adaptors and low quality bases using the program Trimmomatic v.0.35[74](https://www.nature.com/articles/s41467-019-10600-7#ref-CR74) (parameters: ILLUMINACLIP:NexteraPE-PE.fa:2:30:10 CROP:250 HEADCROP:5 LEADING:20 TRAILING:20 SLIDINGWINDOW:4:20 MINLEN:36). As the reference we joined the *E. coli* MG1655 (GenBank accession no. NC_000913.3) genome with the pIND plasmid sequence (created in SnapGene software v.2.4 (GLS Biotech)). For the pCON plasmids we used the validated pCON sequence as a reference. The sequencing reads were mapped to the reference genomes using BWA-MEM v.0.7.5a-r405[75](https://www.nature.com/articles/s41467-019-10600-7#ref-CR75). Mapping statistics were retrieved using BAMStats v.1.25 (https://sourceforge.net/projects/bamstats/files/). Subsequent indexing, local realignment of sequencing reads were performed using PICARD tools, SAMtools v.0.1.19[76](https://www.nature.com/articles/s41467-019-10600-7#ref-CR76),[77](https://www.nature.com/articles/s41467-019-10600-7#ref-CR77) and GATK v.3.677 retaining only paired mapped reads with a minimum mapping quality of 20. SNPs were called using LoFreq v.2.1.2[78](https://www.nature.com/articles/s41467-019-10600-7#ref-CR78) and GATK[77](https://www.nature.com/articles/s41467-019-10600-7#ref-CR77). PCN was inferred from the de BAM files as the ratio of plasmid to chromosomal mean coverage. The coverage distribution was further used to infer large structural variants of the plasmids.
>
>    ref: Wein, T., Hülter, N. F., Mizrahi, I. & Dagan, T. Emergence of plasmid stability under non-selective conditions maintains antibiotic resistance. *Nat Commun* **10**, 2595 (2019).  
>
> 2. qPCR for plasmid copy number
>
>    95°C for 10 minutes prior to storage at -20°C
>
>    ref: Škulj, M. *et al.* Improved determination of plasmid copy number using quantitative real-time PCR for monitoring fermentation processes. *Microb Cell Fact* **7**, 6 (2008).  
>
> 





### Perturbed growth rate condition

<img src="nonlinear plasmid copy number model.assets/image-20220206224622224.png" alt="image-20220206224622224" style="zoom:25%;" />

> M5_L3 down up shift
> Data: sub_model/growth_rate_down_up_shift_20210310.xlsx
> Code: sub_model/gr_updown_shift.py:164

<img src="nonlinear plasmid copy number model.assets/image-20220206224636566.png" alt="image-20220206224636566" style="zoom:25%;" />

> L2 down up shift
> Data: sub_model/growth_rate_down_up_shift_20210310.xlsx
> Code: sub_model/gr_updown_shift.py:164

<img src="nonlinear plasmid copy number model.assets/image-20220209144209987.png" alt="image-20220209144209987" style="zoom:25%;" />

> code source: sub_model/gr_updown_shift.py:262

<img src="nonlinear plasmid copy number model.assets/image-20220209143319719.png" alt="image-20220209143319719" style="zoom:25%;" />

> code source: sub_model/gr_updown_shift.py:247



<img src="nonlinear plasmid copy number model.assets/image-20220209184051953.png" alt="image-20220209184051953" style="zoom:25%;" />

> Code: sub_model/gr_updown_shift.py:283

