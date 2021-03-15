# import io
# import sys
# import numpy as np
# from simulation.hawkes_process import hawkes_process
# from simulation.simulate_data import simulate_data
# from simulation.kernelParameters import kernelParameters
# from simulation.utility import utility
# from scipy.optimize import minimize
# from scipy.optimize import OptimizeResult
# from synthetic_data_processing import synthetic_data_processing
# from scipy.special import erf
# from scipy.optimize import check_grad
# import decimal
#
# from transformation import transformation
#
# # Precision to use
# decimal.getcontext().prec = 100
#
# class gradientAscent:
#
#
#     # def custom_gradient_mu(self, mu, *args):
#     #
#     #     G = np.zeros((self.U, self.U))
#     #     eta = 1.0
#     #     fixed_gradient_mu = 1000.0
#     #     mus = np.zeros((self.I, self.U))
#     #     obj_func_values = []
#     #
#     #     for t in range(0, self.I):
#     #
#     #         if t % 50 == 0:
#     #             print "Iteration: ", t
#     #             print "mu: ", mu
#     #             fixed_gradient_mu = fixed_gradient_mu / 10.0
#     #
#     #         # updated arguments
#     #         args = list(args)
#     #         args[0]['mu'] = mu
#     #         args = tuple(args)
#     #
#     #         # terminating condition
#     #         obj_func_values.append(self.obj_func(args))
#     #         if t != 0 and (np.abs(obj_func_values[t] - obj_func_values[t - 1])) < 0.01:
#     #             print obj_func_values[t]
#     #             print obj_func_values[t - 1]
#     #             break
#     #
#     #         # call to gradient
#     #         grad_estimate = self.gradient_mu(mu, args)
#     #
#     #         G = G + np.outer(grad_estimate, grad_estimate)
#     #
#     #         # gamma = log mu (wrapper to ensure phi is always positive) thus
#     #         # gradient changes from grad_estimate to grad_estimate * phi
#     #
#     #         for i in range(0, len(grad_estimate)):
#     #             grad_estimate[i] = grad_estimate[i] * mu[i]
#     #
#     #         # mu_new = mu + ((eta * 1/np.sqrt(np.diag(G))) * grad_estimate)
#     #         mu_new = mu + fixed_gradient_mu * grad_estimate
#     #         print "new_mu : "
#     #         print mu_new
#     #
#     #         mu = mu_new
#     #
#     #     fout = io.open("obj_func_mu","w")
#     #
#     #     for i in obj_func_values:
#     #         fout.write(unicode(i)+u"\n")
#     #
#     #     fout.close()
#
#     def custom_gradient_theta(self, theta, *args):
#
#         theta_new = np.reshape(theta, (self.V, self.M))
#         theta_trans = np.array(theta_new)
#
#         fixed_step_size = 0.01
#         thetas = np.zeros((self.I, self.U))
#         obj_func_values = []
#
#         for t in range(0, self.I):
#
#             if t % 50 == 0:
#                 print "Iteration: ", t
#                 print "theta: ", theta
#
#             # updated arguments
#             args = list(args)
#             args[0]['theta'] = theta
#             args = tuple(args)
#
#             # terminating condition
#             obj_func_values.append(self.obj_func(args))
#             if t != 0 and (np.abs(obj_func_values[t] - obj_func_values[t - 1])) < 0.01:
#                 print obj_func_values[t]
#                 print obj_func_values[t - 1]
#                 break
#
#             # call to gradient
#             grad_estimate = self.gradient_theta(theta, args)
#
#             # gamma = log theta (wrapper to ensure phi is always positive) thus
#             # gradient changes from grad_estimate to grad_estimate * phi
#
#             for i in range(0, len(grad_estimate)):
#                 grad_estimate[i] = grad_estimate[i] * theta[i]
#
#             theta_new = theta + (fixed_step_size * grad_estimate)
#
#             theta_new = np.reshape(theta_new, (self.V, self.M))
#
#             theta_trans = theta_new.T
#             # print theta_trans
#             theta_trans = utility().normalize_2d_array(theta_trans)
#             theta_new = theta_trans.T
#             theta_new = np.reshape(theta_new, (1, np.product(theta_new.shape)))
#
#             print "new_theta : "
#             print theta_new
#
#             theta = theta_new[0,:]
#
#         fout = io.open("obj_func_theta", "w")
#
#         for i in obj_func_values:
#             fout.write(unicode(i)+u"\n")
#
#         fout.close()
#
#     def custom_gradient_Aij(self, Aij, *args):
#
#         # G = np.zeros(((self.U * self.U), (self.U * self.U)))
#         # eta = 1.0
#
#         fixed_gradient = 0.1
#         Aijs = np.zeros((self.I, self.U * self.U))
#
#         obj_func_values = []
#
#         for t in range(0, self.I):
#
#             Aijs[t] = Aij
#
#             if t % 50 == 0:
#                 print "Iteration: ", t
#                 print "Aij: ", Aij
#
#             # updated arguments
#             args = list(args)
#             args[0]['Aij'] = Aij
#             args = tuple(args)
#
#             # terminating condition
#             obj_func_values.append(self.obj_func(args))
#             if t != 0 and (np.abs(obj_func_values[t] - obj_func_values[t - 1])) < 0.01:
#                 print obj_func_values[t]
#                 print obj_func_values[t - 1]
#                 break
#
#             # call to gradient
#             grad_estimate = self.gradient_A_ij(Aij, args)
#
#             for i in range(0, len(grad_estimate)):
#                 grad_estimate[i] = grad_estimate[i] * Aij[i]
#
#             Aij_new = Aij + (fixed_gradient * grad_estimate)
#
#             print Aij_new
#
#             Aij = Aij_new
#
#         fout = io.open("obj_func_Aij", "w")
#
#         for i in obj_func_values:
#             fout.write(unicode(i)+u"\n")
#
#         fout.close()
#
#
