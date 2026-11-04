library(mgcv)
library(nlme)
library(abind)
library(gratia)
library(ggeffects)
library(emmeans)
library(tidyverse)
library(itsadug)
library(dplyr)

subjects <- c('BM_C03', 'BM_C04', 'BM_C05', 'BM_C07', 'BM_C08', 'BM_C09',
              'BM_C10', 'BM_C12', 'BM_C14', 'BM_C15', 'BM_C16', 'BM_C17',
              'BM_C18', 'BM_C20', 'BM_C22', 'BM_C23', 'BM_C24', 'BM_C25',
              'BM_C26')

df_subj <- data.frame()
for (subj in subjects) {
  print(subj)
  df <- read.csv(paste0('pathfrom', subj, '_SMA-med_inh_allfreqs_gamma_allepo_zscored.csv'))
  df_subj <- rbind(df_subj, df)
}

df_subj$Subject <- as.factor(df_subj$Subject)
df_subj$Freq <- as.factor(df_subj$Freq)

df_subj_filtered <- df_subj %>%
  filter(Inh_time >= -0.07, Inh_time <= 0.075)


model_inh3 <- bam(
  Gamma ~ 
    s(Power_res, by=Freq) + 
    s(Inh_time,k=60) +
    s(Subject, bs = "re") +
    s(Inh_time, Subject, bs="fs")+
    s(Power_res, Subject, bs = "fs"),  
  data = df_subj_filtered,
  method = "fREML"
)
summary(model_inh3)
gam.check(model_inh3)

(ggemmeans(model_inh3, terms = c("Freq [6:30, by=1]", "Power_res [-1.8:1.3, by=0.25]"))) %>% 
  ggplot(aes(x = x, y = predicted, color = as.numeric(as.character(group)), group = group)) +
  geom_line() +
  scale_color_viridis_c() +
  xlab("Frequency") + 
  ylab("Log10 Gamma") +
  labs(color = "Power (residuals)")


# --- Color palette (parula → subset) ---
full_colors <- pals::parula(100)
subset_colors <- full_colors[1:90]


# --- Compute predictions ---
p_freq <- ggemmeans(
  model_inh3,
  terms = c("Freq [6:30, by=1]",
            "Power_res [-1.8:1.3, by=0.2]"))

# --- Convert to numeric explicitly ---
p_freq$x     <- as.numeric(as.character(p_freq$x))
p_freq$group <- as.numeric(as.character(p_freq$group))

# --- Styled Frequency plot ---
p <- ggplot(p_freq, aes(x = x,
                        y = predicted,
                        color = group,
                        group = group)) +
  
  geom_line(size = 1.2) +
  
  # Axis ticks
  scale_x_continuous(breaks = seq(6, 30, by = 2)) +
  scale_y_continuous(breaks = seq(-16.35, -16.00, by = 0.025)) +
  
  # Color scale
  scale_color_gradientn(
    colors = subset_colors,
    name = "Power (residuals)"
  ) +
  
  # Labels
  xlab("Frequency (Hz)") +
  ylab("Log10 Gamma") +
  
  # Matplotlib-like theme
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

ggsave("beta_vs_gamma_overfreqs_4SMA_noman.svg", plot = p, width = 8.5, height = 4.5, units = "in")