library(mgcv)
library(nlme)
library(abind)
library(gratia)
library(ggeffects)
library(emmeans)
library(tidyverse)
library(itsadug)
library(dplyr)
library(pals)

subjects <- c('BM_C03', 'BM_C04', 'BM_C05', 'BM_C07', 'BM_C08', 'BM_C09',
              'BM_C10', 'BM_C12', 'BM_C14', 'BM_C15', 'BM_C16', 'BM_C17',
              'BM_C18', 'BM_C20', 'BM_C22', 'BM_C23', 'BM_C24', 'BM_C25',
              'BM_C26')

#AUTOMATIC DETECTION
df_subj <- data.frame()
for (subj in subjects) {
  print(subj)
  df <- read.csv(paste0('/for_r/', subj, '_gammaburst_vs_behav_auto.csv'))
  df_subj <- rbind(df_subj, df)
}
df_subj$Subject <- as.factor(df_subj$Subject)
df_subj$Burst <- as.factor(df_subj$Burst)

df_subj_filtered_burst1 <- df_subj %>%
  filter(Inh_time>-0.07, Inh_time <= 0.15, Burst==1)

df_subj_filtered_burst2 <- df_subj %>%
  filter(Inh_time>-0.07, Inh_time <= 0.15, Burst==2)
df_subj_filtered_all <- df_subj %>%
  filter(Inh_time>-0.07, Inh_time <= 0.15)

#gamma vs behav: auto
model_inh3 <- bam(Gamma ~ Inh_ampl + Inh_time + Inh_incr +
                    s(Subject, Inh_ampl, bs='re') + s(Subject, Inh_time, bs='re') + s(Subject, Inh_incr, bs='re') +
                    s(Subject, bs='re'), 
                  data = df_subj_filtered_burst1,
                  method = 'fREML')
summary(model_inh3)
svg("gamma_vs_inhampl_burst1_main.svg", width = 6.5, height = 5.5)
plot_smooth(model_inh3, view = "Inh_ampl", rug = TRUE,
            xlab = "Peak EMG Inhibition",
            ylab = "High-gamma power")
dev.off()


model_inh3 <- bam(Gamma ~ Inh_ampl + Inh_time + Inh_incr +
                    s(Subject, Inh_ampl, bs='re') + s(Subject, Inh_time, bs='re') + s(Subject, Inh_incr, bs='re') +
                    s(Subject, bs='re'), 
                  data = df_subj_filtered_burst2,
                  method = 'fREML')
summary(model_inh3)

svg("gamma_vs_inhincr_burst2_main.svg", width = 6.5, height = 5.5)
plot_smooth(model_inh3, view = "Inh_incr", rug = TRUE,
            xlab = "EMG rebound",
            ylab = "High-gamma power")
dev.off()

#MANUAL DETECTION
df_subj <- data.frame()
for (subj in subjects) {
  print(subj)
  df <- read.csv(paste0('/for_r/', subj, '_gammaburst_vs_behav_man.csv'))
  df_subj <- rbind(df_subj, df)
}
df_subj_filtered_burst1 <- df_subj %>%
  filter(Inh_onset>-0.1, Inh_onset <= 0.1, Burst==1)
df_subj_filtered_burst2 <- df_subj %>%
  filter(Inh_onset>-0.1, Inh_onset <= 0.1, Burst==2)

#gamma vs behav: man
model_inh3 <- bam(Gamma ~ Inh_onset + Inh_dur +
                    s(Subject, Inh_dur, bs='re') + s(Subject, Inh_onset, bs='re') +
                    s(Subject, bs='re'),
                  data = df_subj_filtered_burst1,
                  method = 'fREML')
summary(model_inh3)

model_inh3 <- bam(Gamma ~ Inh_onset + Inh_dur +
                    s(Subject, Inh_dur, bs='re') + s(Subject, Inh_onset, bs='re') +
                    s(Subject, bs='re'), 
                  data = df_subj_filtered_burst2,
                  method = 'fREML')
summary(model_inh3)
plot_smooth(model_inh3, view = "Inh_onset", rug = TRUE)

svg("gamma_vs_inhonset_burst2_main.svg", width = 6.5, height = 5.5)
plot_smooth(model_inh3, view = "Inh_onset", rug = TRUE,
            xlab = "EMG inhibition onset, s",
            ylab = "High-gamma power")
dev.off()



#ELBOW ROTATION PARAMETERS
df_subj <- data.frame()
for (subj in subjects) {
  print(subj)
  df <- read.csv(paste0('/for_r/', subj, '_gammaburst_vs_behav_rot.csv'))
  df_subj <- rbind(df_subj, df)
}
df_subj_filtered_burst1 <- df_subj %>%
  filter(Burst==1)
df_subj_filtered_burst2 <- df_subj %>%
  filter(Burst==2)

#gamma vs behav: rot
model_inh3 <- bam(Gamma ~ Peak_rot + Rot_decl +
                    s(Subject, Peak_rot, bs='re') + s(Subject, Rot_decl, bs='re') +
                    s(Subject, bs='re'), 
                  data = df_subj_filtered_burst1,
                  method = 'fREML')
summary(model_inh3)

model_inh3 <- bam(Gamma ~ Peak_rot + Rot_decl +
                    s(Subject, Peak_rot, bs='re') + s(Subject, Rot_decl, bs='re') +
                    s(Subject, bs='re'), 
                  data = df_subj_filtered_burst2,
                  method = 'fREML')
summary(model_inh3)

svg("gamma_vs_peakrot_burst2_main.svg", width = 6.5, height = 5.5)
plot_smooth(model_inh3, view = "Peak_rot", rug = TRUE,
            xlab = "Peak Elbow Rotation, s",
            ylab = "High-gamma power")
dev.off()