/* Model for joint constrtains from CO auto, Galaxy auto and CO X A power spectra */
data{
    int N; // Numebr of k modes
    real n3D; // 3D number density of the galaxies, to be used for the galaxy shot noise
    vector[N] pm; // Linear matter power spetrum in real space
    vector[N] lim_pk; // Observed CO auto power spectrum
    vector[N] gal_pk; // Observed auto power spectrum for field A
    vector[N] lim_gal_pk; // Observed CO X Lya power spectrum
    vector[N] sigma_lim_pk; // The uncertainty in the co power, i.e. sigma_P(k).
                        // Needs more thought Issue #60
    vector[N] sigma_gal_pk; // The uncertainty in the A auto power, i.e. sigma_P(k).
                        // Needs more thought Issue #60
    vector[N] sigma_lim_gal_pk; // The uncertainty in the cross power
}

parameters { 
    real <lower=0> bgal;
    real <lower=0> clustering;
    real pshot_lim;
    real pshot_gal;
    real pshot_lim_gal;
}

model {
    /* Likliehood  */
    gal_pk  ~ normal(pm*bgal*bgal + pshot_gal, sigma_gal_pk);
    lim_pk ~ normal(pm*clustering*clustering + pshot_lim, sigma_lim_pk);
    lim_gal_pk ~ normal(pm*clustering*bgal + pshot_lim_gal, sigma_lim_gal_pk);
}
