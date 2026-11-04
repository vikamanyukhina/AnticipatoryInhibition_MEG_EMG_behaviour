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

subjects <- paste0('BM_C', sprintf('%02d', setdiff(3:27, c(13))))


df_subj <- data.frame()
for (subj in subjects) {
  print(subj)
  df <- read.csv(paste0('/for_r/', subj, '_SMA-med_test_zscored_new.csv'))
  df_subj <- rbind(df_subj, df)
}
df_subj$Subject <- as.factor(df_subj$Subject)

df_subj_filtered <- df_subj %>%
  filter(Inh_time>-0.07, Inh_time <= 0.15)



#SMA GAMMA 90-130Hz VS INHIBITION AMPLITUDE
model_inh3 <- bam(Inh ~ s(Gamma) + s(Inh_time) +
                    ti(Gamma,Inh_time) +
                    s(Subject, Gamma, bs='re') + s(Subject, Inh_time, bs='re') +
                    s(Subject, bs='re'), 
                  data = df_subj_filtered,
                  method = 'fREML')
summary(model_inh3)
plot_smooth(model_inh3, view = "Inh_time", rug = TRUE)
plot_smooth(model_inh3, view = "Gamma", rug = TRUE)
(ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.025]","Gamma [-17.7:-15.1, by=0.25]"))) %>% ggplot(aes(x = x, y = predicted, ymin=conf.low, ymax=conf.high, color=as.numeric(as.character(group)), group = group)) +     geom_line() + scale_color_viridis_c()+ xlab('Inh time, s') + ylab('Inh')  +labs(color = "Gamma")

# Generate predicted values for the smooth term
pred <- ggemmeans(model_inh3, terms = "Inh_time")

# Plot
p <- (ggplot(pred, aes(x = x, y = predicted)) +
        geom_line(color = "black", size = 1) +
        geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
        geom_rug(data = df_subj_filtered, aes(x = Inh_time), inherit.aes = FALSE) +
        labs(x = "Inhibition Peak Time, s",
             y = "Predicted Peak Inhibition") +
        theme_minimal(base_size = 16))

plot(p)  # built-in method draws ggplot
ggsave("gam_gamma_vs_inh_inhtime_maineffect.svg", plot = p, width = 6, height = 4, units = "in")

#MAKE A PLOT
full_colors <- parula(100)

# keep only the last ~50% of the colors (orange → red)
subset_colors <- full_colors[0:90]

# Now use in ggplot
p <- (ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.025]",
                                      "Gamma [-17.7:-15.1, by=0.25]"))) %>%
  ggplot(aes(x = x,
             y = predicted,  # use -predicted if you want stronger inhibition negative
             color = as.numeric(as.character(group)),
             group = group)) +
  geom_line(size = 1.2) +
  
  # Color scale
  scale_color_gradientn(
    colors = subset_colors,
    name = "Log10 Gamma power"
  ) +
  
  # Labels
  xlab("Inhibition time, s (0-unloading)") +
  ylab("Peak EMG Inhibition") +
  scale_y_continuous(limits = c(0, 1.75)) +
  
  # Matplotlib-style theme
  theme_bw(base_size = 19, base_family = "DejaVu Sans") +
  theme(
    panel.grid.major = element_line(color = "grey90", size = 0.5),  # subtle grid
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 15),
    axis.title = element_text(size = 19),
    legend.background = element_blank(),
    legend.key = element_blank(),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 15)
  )
plot(p)  # built-in method draws ggplot
ggsave("gamma_vs_inh_4SMA.svg", plot = p, width = 8, height = 4, units = "in")


#SMA BETA 19-24Hz VS INHIBITION AMPLITUDE

model_inh3 <- bam(Inh ~ s(Beta_early1_res) + s(Inh_time) +
                    ti(Beta_early1_res,Inh_time) +
                    s(Subject, Beta_early1_res, bs='re') + s(Subject, Inh_time, bs='re') +
                    s(Subject, bs='re'), 
                  data = df_subj_filtered,
                  method = 'fREML')
summary(model_inh3)
plot_smooth(model_inh3, view = "Inh_time", rug = TRUE)
plot_smooth(model_inh3, view = "Beta_early1_res", rug = TRUE)
(ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.025]","Beta_early1_res [-1:1, by=0.15]"))) %>% ggplot(aes(x = x, y = predicted, ymin=conf.low, ymax=conf.high, color=as.numeric(as.character(group)), group = group)) +     geom_line() + scale_color_viridis_c()+ xlab('Inh time, s') + ylab('Inh')  +labs(color = "Beta")


#MAKE A PLOT
full_colors <- parula(100)

# keep only the last ~50% of the colors (orange → red)
subset_colors <- full_colors[0:90]

# Now use in ggplot
p <- (ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.025]",
                                      "Beta_early1_res [-1:1, by=0.25]"))) %>%
  ggplot(aes(x = x,
             y = predicted,  # use -predicted if you want stronger inhibition negative
             color = as.numeric(as.character(group)),
             group = group)) +
  geom_line(size = 1.2) +
  
  # Color scale
  scale_color_gradientn(
    colors = subset_colors,
    name = "Log10 Beta 19-24Hz"
  ) +
  
  # Labels
  xlab("Inhibition time, s (0-unloading)") +
  ylab("Peak EMG Inhibition") +
  scale_y_continuous(limits = c(0.25, 1.5)) +
  
  # Matplotlib-style theme
  theme_bw(base_size = 19, base_family = "DejaVu Sans") +
  theme(
    panel.grid.major = element_line(color = "grey90", size = 0.5),  # subtle grid
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 15),
    axis.title = element_text(size = 19),
    legend.background = element_blank(),
    legend.key = element_blank(),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 15)
  )
plot(p)  # built-in method draws ggplot
ggsave("beta1_vs_inh_4SMA.svg", plot = p, width = 8, height = 4, units = "in")


#SMA BETA 25-29Hz VS INHIBITION AMPLITUDE

model_inh3 <- bam(Inh ~ s(Beta_early2_res) + s(Inh_time) +
                    ti(Beta_early2_res,Inh_time) +
                    s(Subject, Beta_early2_res, bs='re') + s(Subject, Inh_time, bs='re') +
                    s(Subject, bs='re'), 
                  data = df_subj_filtered,
                  method = 'fREML')
summary(model_inh3)
plot_smooth(model_inh3, view = "Beta_early2_res", rug = TRUE)
(ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.025]","Beta_early2_res [-1.17:0.92, by=0.1]"))) %>% ggplot(aes(x = x, y = predicted, ymin=conf.low, ymax=conf.high, color=as.numeric(as.character(group)), group = group)) +     geom_line() + scale_color_viridis_c()+ xlab('Inh time, s') + ylab('Inh')  +labs(color = "Beta")

preds <- ggemmeans(
  model_inh3,
  terms = c("Inh_time [-0.07:0.15, by=0.025]",
            "Beta_IFC_res [-1.17:0.92, by=0.15]")
)

p <- preds %>%
  ggplot(aes(
    x = x, 
    y = predicted,  # or -predicted if stronger inhibition is negative
    color = as.numeric(as.character(group)),
    group = group
  )) +
  geom_line(size = 1.2) +
  
  # Color gradient
  scale_color_gradientn(
    colors = subset_colors,
    name = "Log10 Beta 25-29Hz"
  ) +
  
  # Axis labels
  xlab("Inhibition time, s (0 = unloading)") +
  ylab("Peak EMG Inhibition") +
  scale_y_continuous(limits = c(0, 2)) +
  
  # Theme: clean, Matplotlib-style
  theme_bw(base_size = 19, base_family = "DejaVu Sans") +
  theme(
    panel.grid.major = element_line(color = "grey90", size = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 15),
    axis.title = element_text(size = 19),
    legend.background = element_blank(),
    legend.key = element_blank(),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 15)
  )
p <- p + scale_y_reverse(limits = c(2, 0))
plot(p)
ggsave("beta2_vs_inh_4SMA.svg", plot = p, width = 8, height = 4, units = "in")


#IFC BETA 14-19Hz VS INHIBITION AMPLITUDE
model_inh3 <- bam(Inh ~ s(Beta_IFC_res) + s(Inh_time) +
                    ti(Beta_IFC_res,Inh_time) +
                    s(Subject, Beta_IFC_res, bs='re') + s(Subject, Inh_time, bs='re') +
                    s(Subject, bs='re'), 
                  data = df_subj_filtered,
                  method = 'fREML')
summary(model_inh3)
(ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.025]","Beta_IFC_res [-0.7:0.65, by=0.025]"))) %>% ggplot(aes(x = x, y = predicted, ymin=conf.low, ymax=conf.high, color=as.numeric(as.character(group)), group = group)) +     geom_line() + scale_color_viridis_c()+ xlab('Inh time, s') + ylab("Inh")  +labs(color = "Beta")

#for paper
full_colors <- parula(100)
# keep only the last ~50% of the colors (orange → red)
subset_colors <- full_colors[0:90]
# Now use in ggplot
p <- (ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.025]",
                                      "Beta_IFC_res [-0.7:0.65, by=0.1]"))) %>%
  ggplot(aes(x = x,
             y = predicted,  # use -predicted if you want stronger inhibition negative
             color = as.numeric(as.character(group)),
             group = group)) +
  geom_line(size = 1.2) +
  
  # Color scale
  scale_color_gradientn(
    colors = subset_colors,
    name = "Log10 Beta 14-19Hz"
  ) +
  
  # Labels
  xlab("Inhibition time, s (0-unloading)") +
  ylab("Peak EMG Inhibition") +
  scale_y_continuous(limits = c(0.25, 1.5)) +
  
  # Matplotlib-style theme
  theme_bw(base_size = 19, base_family = "DejaVu Sans") +
  theme(
    panel.grid.major = element_line(color = "grey90", size = 0.5),  # subtle grid
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 15),
    axis.title = element_text(size = 19),
    legend.background = element_blank(),
    legend.key = element_blank(),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 15)
  )
plot(p)  # built-in method draws ggplot
ggsave("IFCbeta_vs_inh_14-19Hz.svg", plot = p, width = 8, height = 4, units = "in")


#M1 GAMMA 90-130Hz (from gamma power contrast vs baseline) 
#VS INHIBITION PEAK TIME: an effect of inhibition time
model_inh3 <- bam(Inh ~ s(Gamma_m1) + s(Inh_time) +
                    ti(Gamma_m1,Inh_time) +
                    s(Subject, Gamma_m1, bs='re') + s(Subject, Inh_time, bs='re') +
                    s(Subject, bs='re'), 
                  data = df_subj_filtered,
                  method = 'fREML')
summary(model_inh3)
plot_smooth(model_inh3, view = "Inh_time", rug = TRUE)
plot_smooth(model_inh3, view = "Gamma_m1", rug = TRUE)

svg("gammacontrast_vs_inhampl_vPMC_main.svg", width = 6.5, height = 5.5)
plot_smooth(model_inh3, view = "Gamma_m1", rug = TRUE,
            xlab = "High-gamma power",
            ylab = "Peak EMG Inhibition")
dev.off()


#BETA 23-26Hz VS INHIBITION DURATION (manual detection)

df_subj <- data.frame()
for (subj in subjects) {
  print(subj)
  df <- read.csv(paste0('/for_r/', subj, '_SMA-med_test_zscored_man.csv'))
  df_subj <- rbind(df_subj, df)
}
df_subj$Subject <- as.factor(df_subj$Subject)

df_subj_filtered <- df_subj %>%
  filter(Inh_time>-0.07, Inh_time <= 0.15)

model_inh3 <- bam(Inh_dur ~ s(Beta_early2_res) + s(Inh_time) +
                    ti(Beta_early2_res,Inh_time) +
                    s(Subject, Beta_early2_res, bs='re') + s(Subject, Inh_time, bs='re') +
                    s(Subject, bs='re'), 
                  data = df_subj_filtered,
                  method = 'fREML')
summary(model_inh3)
plot_smooth(model_inh3, view = "Beta_early2_res", rug = TRUE)
plot_smooth(model_inh3, view = "Inh_time", rug = TRUE)
(ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.025]","Beta_early2_res [-1.06:1.1, by=0.15]"))) %>% ggplot(aes(x = x, y = predicted, ymin=conf.low, ymax=conf.high, color=as.numeric(as.character(group)), group = group)) +     geom_line() + scale_color_viridis_c()+ xlab('Inh time, s') + ylab('Inh')  +labs(color = "Beta")


# Generate predicted values for the smooth term
pred <- ggemmeans(model_inh3, terms = "Inh_time")

# Plot
p <- (ggplot(pred, aes(x = x, y = predicted)) +
        geom_line(color = "black", size = 1) +
        geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
        geom_rug(data = df_subj_filtered, aes(x = Inh_time), inherit.aes = FALSE) +
        labs(x = "Inhibition Peak Time, s",
             y = "Predicted Inhibition Duration") +
        theme_minimal(base_size = 16))

plot(p)  # built-in method draws ggplot
ggsave("gam_inhduration_vs_inh_inhtime_maineffect.svg", plot = p, width = 6, height = 4, units = "in")