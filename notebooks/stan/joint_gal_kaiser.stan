/* Model for joint constrtains from CO auto, Galaxy auto and CO X A power spectra */
data{
    int N; // Numebr of k modes
    real n3D; // 3D number density of the galaxies, to be used for the galaxy shot noise
    vector[N] pm; // Linear matter power spetrum in real space
    vector[N] co_pk; // Observed CO auto power spectrum
    vector[N] gal_pk; // Observed auto power spectrum for field A
    vector[N] co_gal_pk; // Observed CO X Lya power spectrum
    vector[N] sigma_co_pk; // The uncertainty in the co power, i.e. sigma_P(k).
                        // Needs more thought Issue #60
    vector[N] sigma_gal_pk; // The uncertainty in the A auto power, i.e. sigma_P(k).
                        // Needs more thought Issue #60
    vector[N] sigma_co_gal_pk; // The uncertainty in the cross power
}

parameters { 
    real <lower=0> bgal;
    real <lower=0> Tbco;
    real <lower=0> bco;
    real pshot_co;
    real pshot_gal;
    real pshot_co_gal;
}

model {
    /* Likliehood  */
    gal_pk  ~ normal(pm*(bgal*bgal+ (2/3)*bgal + 1/5) + pshot_gal, sigma_gal_pk);
    co_pk ~ normal(pm*Tbco*Tbco + pshot_co, sigma_co_pk);
    co_gal_pk ~ normal(pm*Tbco*bgal*(1 + 1/(3*bgal) + 1/(3*bco) + 1/(5*bgal*bco)) + pshot_co_gal, sigma_co_gal_pk);
}
