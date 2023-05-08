/* Model for joint constrtains from CO auto, Lya auto and CO X Lya*/
data{
    int N; // Numebr of k modes
    vector[N] pm; // Linear matter power spetrum in real space
    vector[N] lim_pk; // Observed CO auto power spectrum
    vector[N] lya_pk; // Observed auto power spectrum for field A
    vector[N] lim_lya_pk; // Observed CO X Lya power spectrum
    vector[N] sigma_lim_pk; // The uncertainty in the co power, i.e. sigma_P(k).
                        // Needs more thought Issue #60
    vector[N] sigma_lya_pk; // The uncertainty in the A auto power, i.e. sigma_P(k).
                        // Needs more thought Issue #60
    vector[N] sigma_lim_lya_pk; // The uncertainty in the cross power, 
                            //i.e. sigma_P(k). Needs more thought Issue #60
}

parameters { 
    real <upper=0> blya;
    real clustering;
    real pshot_lim;
}

model {
    /* Likliehood  */
    lya_pk  ~ normal(pm*blya*blya, sigma_lya_pk);
    lim_pk ~ normal(pm*clustering*clustering + pshot_lim, sigma_lim_pk);
    lim_lya_pk ~ normal(-pm*clustering*blya, sigma_lim_lya_pk);
}
