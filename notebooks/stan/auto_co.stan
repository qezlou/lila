/* Model for any auto spectrum */
data{
    int N; // Numebr of k modes
    vector[N] pm; // matter power spetrum
    vector[N] lim_pk; // Observed CO power spectrum
    vector[N] sigma_lim_pk; // The uncertainty in the CO power, i.e. sigma_P(k). Needs more thought Issue #60
}

parameters { 
    real <lower=0> clustering;
    real pshot_lim;
}

model {
    /* Likliehood  */
    lim_pk ~ normal(pm*clustering*clustering + pshot_lim, sigma_lim_pk);
}
