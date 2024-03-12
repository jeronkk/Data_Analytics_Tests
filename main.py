import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def two_sided_t_test(data, alpha, u_bar):
    x_bar = np.mean(data)
    std = np.std(data,ddof=1)
    x_n = len(data)
    t = (x_bar - u_bar)/(std/np.sqrt(x_n))
    t_crit = stats.t.ppf(1 - alpha/2,x_n-1)
    
    print("T-stat = ",t,"\nT-crit = ",t_crit)
    if (np.abs(t) > t_crit):
        print("2-Sided: Reject null hypothesis")
    else:
        print("2-Sided: Don't reject null hypothesis")
    
    p_val = 2*(1 - stats.t.cdf(np.abs(t),x_n-1))
    print("P-value = ",p_val,"\n")

def one_sided_t_test(data, alpha, u_bar):
    x_bar = np.mean(data)
    std = np.std(data,ddof=1)
    x_n = len(data)
    t = (x_bar - u_bar)/(std/np.sqrt(x_n))
    t_crit = stats.t.ppf(1 - alpha,x_n-1)
    
    print("T-stat = ",t,"\nT-crit = ",t_crit)
    if (t > t_crit):
        print("1-Sided: Reject null hypothesis")
    else:
        print("1-Sided: Don't reject null hypothesis")

    p_val = 1 - stats.t.cdf(t,n - 1)
    print("P-value = ",p_val,"\n")

def confidence_interval_pop_std_known(data, pop_std, alpha):
    x_bar = np.mean(data)
    n = len(data)
    t_crit = stats.t.ppf(1 - alpha/2,n - 1)
    lower = x_bar - t_crit*pop_std/np.sqrt(n)
    upper = x_bar + t_crit*pop_std/np.sqrt(n)
    print(lower," < μ < ",upper,"\n")

def confidence_interval_pop_std_unknown(data, alpha):
    x_bar = np.mean(data)
    std = np.std(data,ddof=1)
    n = len(data)
    t_crit = stats.t.ppf(1 - alpha/2,n - 1)
    lower = x_bar - t_crit*std/np.sqrt(n)
    upper = x_bar + t_crit*std/np.sqrt(n)
    print(lower," < μ < ",upper,"\n")

def two_mean_t_test_pop_std_known(data1, data2, alpha, pop_std, u_diff):
    x_bar = np.mean(data1)
    y_bar = np.mean(data2)
    x_n = len(data1)
    y_n = len(data2)
    t = ((x_bar - y_bar) - u_diff)/pop_std*np.sqrt(1/x_n + 1/y_n)
    t_crit = stats.t.ppf(1 - alpha/2,x_n+y_n-2)
    
    print("T-stat = ",t,"\nT-crit = ",t_crit)
    if (np.abs(t) > t_crit):
        print("2-Mean, POP_STD Known: Reject null hypothesis")
    else:
        print("2-Mean, POP_STD Known: Don't reject null hypothesis")
    
    p_val = 2*(1 - stats.t.cdf(np.abs(t),x_n+y_n-2))
    print("P-value = ",p_val,"\n")

def two_mean_t_test_pop_std_unknown(data1, data2, alpha, u_diff):
    x_bar = np.mean(data1)
    y_bar = np.mean(data2)
    x_std = np.std(data1,ddof=1)
    y_std = np.std(data2,ddof=1)
    x_n = len(data1)
    y_n = len(data2)
    sp = np.sqrt(((x_n-1)*x_std**2 + (y_n-1)*y_std**2)/(x_n + y_n - 2))
    t = ((x_bar - y_bar) - u_diff)/(sp*np.sqrt(1/x_n + 1/y_n))
    t_crit = stats.t.ppf(1 - alpha/2,x_n+y_n-2)
    
    print("T-stat = ",t,"\nT-crit = ",t_crit)
    if (np.abs(t) > t_crit):
        print("2-Mean, POP_STD Unknown: Reject null hypothesis")
    else:
        print("2-Mean, POP_STD Unknown: Don't reject null hypothesis")
    
    p_val = 2*(1 - stats.t.cdf(np.abs(t),x_n+y_n-2))
    print("P-value = ",p_val,"\n")

def two_mean_t_test_unequal_var(data1, data2, alpha, u_diff):
    x_bar = np.mean(data1)
    y_bar = np.mean(data2)
    x_std = np.std(data1,ddof=1)
    y_std = np.std(data2,ddof=1)
    x_n = len(data1)
    y_n = len(data2)
    t = ((x_bar - y_bar) - u_diff)/np.sqrt((x_std**2/x_n) + (y_std**2/y_n))
    dof_x = x_std**2/x_n
    dof_y = y_std**2/y_n
    dof = ((dof_x +dof_y)**2)/((dof_x**2)/(x_n - 1) + (dof_y**2)/(y_n - 1))
    t_crit = stats.t.ppf(1 - alpha/2,round(dof))
    
    print("T-stat = ",t,"\nT-crit = ",t_crit)
    if (np.abs(t) > t_crit):
        print("2-Mean, Unequal Var: Reject null hypothesis")
    else:
        print("2-Mean, Unequal Var: Don't reject null hypothesis")
    
    p_val = 2*(1 - stats.t.cdf(np.abs(t),round(dof)))
    print("P-value = ",p_val,"\n")

def power_of_t_test(data1, data2, pop_std, diff):
    x_n = len(data1)
    y_n = len(data2)
    cod = diff/(pop_std*np.sqrt(1/x_n + 1/y_n))
    t_star = t_crit - cod
    power = 1 - stats.t.cdf(np.abs(t_star),x_n + y_n - 2)
    print("Power = ",power,"\n")

def paired_t_test(data1, data2, hypo_diff):
    diff = data1 - data2
    diff_mean = np.mean(diff)
    diff_std = np.std(diff,ddof=1)
    diff_n = len(diff)
    t = (diff_mean - hypo_diff)/(diff_std/np.sqrt(diff_n))
    t_crit = stats.t.ppf(0.995,diff_n - 1)
    print("T-stat = ",t,"\nT-crit = ",t_crit)
    if (np.abs(t) > t_crit):
        print("Paired T-test: Reject null hypothesis")
    else:
        print("Paired T-test: Don't reject null hypothesis")
    p_val = 2*(1 - stats.t.cdf(np.abs(t),diff_n - 1))
    print("P-value = ",p_val,"\n")

def bonferroni_test(data1, data2, alpha, s_wit, m, total_n):
    x_bar = np.mean(data1)
    y_bar = np.mean(data2)
    x_n = len(data1)
    y_n = len(data2)
    t = (x_bar - y_bar)/np.sqrt(s_wit/x_n + s_wit/y_n)
    if (x_n == y_n):    
        t_crit = stats.t.ppf(1 - alpha/2,m*(x_n - 1))
    else: 
        t_crit = stats.t.ppf(1 - alpha/2,total_n - m)    
    if (np.abs(t) > t_crit):
        print("PAIRWISE: Reject null hypothesis")
    else:
        print("PAIRWISE: Don't reject null hypothesis")

def anova_4(data1, data2, data3, data4, alpha):
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    mean3 = np.mean(data3)
    mean4 = np.mean(data4)
    var1 = np.var(data1,ddof=1)
    var2 = np.var(data2,ddof=1)
    var3 = np.var(data3,ddof=1)
    var4 = np.var(data4,ddof=1)
    s_wit = (var1+var2+var3+var4)/4
    s_bet = len(data1) * np.var([mean1, mean2, mean3, mean4],ddof=1)
    f = s_bet/s_wit
    f_crit = stats.f.ppf(1 - alpha,3, 4*(len(data1) - 1))
    p_val = 1 - stats.f.cdf(f,3, 4*(len(data1) - 1))
    print("F-stat = ",f,"\nF-crit = ",f_crit,"\nP-value = ",p_val)
    if (f > f_crit):
        print("ANOVA_4: Reject null hypothesis")
        alpha_k = alpha/6
        bonferroni_test(data1,data2,alpha_k,s_wit,4,0)
        bonferroni_test(data1,data3,alpha_k,s_wit,4,0)
        bonferroni_test(data1,data4,alpha_k,s_wit,4,0)
        bonferroni_test(data2,data3,alpha_k,s_wit,4,0)
        bonferroni_test(data2,data4,alpha_k,s_wit,4,0)
        bonferroni_test(data3,data4,alpha_k,s_wit,4,0)
        print("\n")
    else:
        print("ANOVA_4: Don't reject null hypothesis\n")

def anova_3(data1, data2, data3, alpha):
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    mean3 = np.mean(data3)
    var1 = np.var(data1,ddof=1)
    var2 = np.var(data2,ddof=1)
    var3 = np.var(data3,ddof=1)
    s_wit = (var1+var2+var3)/3
    s_bet = len(data1) * np.var([mean1, mean2, mean3],ddof=1)
    f = s_bet/s_wit
    f_crit = stats.f.ppf(1 - alpha,2, 3*(len(data1) - 1))
    p_val = 1 - stats.f.cdf(f,2, 3*(len(data1) - 1))
    print("F-stat = ",f,"\nF-crit = ",f_crit,"\nP-value = ",p_val)
    if (f > f_crit):
        print("ANOVA_3: Reject null hypothesis")
        alpha_k = alpha/3
        bonferroni_test(data1,data2,alpha_k,s_wit,4,0)
        bonferroni_test(data1,data3,alpha_k,s_wit,4,0)
        bonferroni_test(data2,data3,alpha_k,s_wit,4,0)
        print("\n")
    else:
        print("ANOVA_3: Don't reject null hypothesis\n")

def anova_4_unequal(data1, data2, data3, data4, alpha):
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    mean3 = np.mean(data3)
    mean4 = np.mean(data4)
    var1 = np.var(data1,ddof=1)
    var2 = np.var(data2,ddof=1)
    var3 = np.var(data3,ddof=1)
    var4 = np.var(data4,ddof=1)
    n = len(data1)+len(data2)+len(data3)+len(data4)
    sum_x = len(data1)*(mean1)+len(data2)*(mean2)+len(data3)*(mean3)+len(data4)*(mean4)
    sum_x2 = len(data1)*(mean1**2)+len(data2)*(mean2**2)+len(data3)*(mean3**2)+len(data4)*(mean4**2)
    ss_bet = sum_x2 - ((sum_x**2)/n)
    ss_wit = (len(data1)-1)*var1+(len(data2)-1)*var2+(len(data3)-1)*var3+(len(data4)-1)*var4
    s_wit = ss_wit/(n-4)
    s_bet = ss_bet/3
    f = s_bet/s_wit
    f_crit = stats.f.ppf(1 - alpha,3,n-4)
    p_val = 1 - stats.f.cdf(f,3,n-4)
    print("F-stat = ",f,"\nF-crit = ",f_crit,"\nP-value = ",p_val)
    if (f > f_crit):
        print("ANOVA_4_unequal: Reject null hypothesis")
        alpha_k = alpha/6
        bonferroni_test(data1,data2,alpha_k,s_wit,4,0)
        bonferroni_test(data1,data3,alpha_k,s_wit,4,0)
        bonferroni_test(data1,data4,alpha_k,s_wit,4,0)
        bonferroni_test(data2,data3,alpha_k,s_wit,4,0)
        bonferroni_test(data2,data4,alpha_k,s_wit,4,0)
        bonferroni_test(data3,data4,alpha_k,s_wit,4,0)
        print("\n")
    else:
        print("ANOVA_4_unequal: Don't reject null hypothesis\n")

def anova_3_unequal(data1, data2, data3, alpha):
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    mean3 = np.mean(data3)
    var1 = np.var(data1,ddof=1)
    var2 = np.var(data2,ddof=1)
    var3 = np.var(data3,ddof=1)
    n = len(data1)+len(data2)+len(data3)
    sum_x = len(data1)*(mean1)+len(data2)*(mean2)+len(data3)*(mean3)
    sum_x2 = len(data1)*(mean1**2)+len(data2)*(mean2**2)+len(data3)*(mean3**2)
    ss_bet = sum_x2 - ((sum_x**2)/n)
    ss_wit = (len(data1)-1)*var1+(len(data2)-1)*var2+(len(data3)-1)*var3
    s_wit = ss_wit/(n-3)
    s_bet = ss_bet/2
    f = s_bet/s_wit
    f_crit = stats.f.ppf(1 - alpha,2,n-3)
    p_val = 1 - stats.f.cdf(f,2,n-3)
    print("F-stat = ",f,"\nF-crit = ",f_crit,"\nP-value = ",p_val)
    if (f > f_crit):
        print("ANOVA_3_unequal: Reject null hypothesis")
        alpha_k = alpha/3
        bonferroni_test(data1,data2,alpha_k,s_wit,4,0)
        bonferroni_test(data1,data3,alpha_k,s_wit,4,0)
        bonferroni_test(data2,data3,alpha_k,s_wit,4,0)
        print("\n")
    else:
        print("ANOVA_3_unequal: Don't reject null hypothesis\n")

