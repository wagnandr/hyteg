from matplotlib import pyplot as plt
  
'''
it_cell_gs = 33
it_none = 98
it_ilu_inplace = 3
it_ilu_nocorr = [16, 15, 10, 7, 6, 5, 5, 4, 4, 4, 4, 4, 3]
it_ilu_corr = [18, 16, 10, 8, 6, 5, 5, 4, 4, 4, 4, 4, 4]
'''

'''
it_cell_gs = 67
it_ilu_inplace = 6
it_ilu_nocorr = [36, 33, 19, 14, 11, 10, 9, 8, 8, 8, 8, 7, 7]
it_ilu_corr = [41, 34, 20, 14, 11, 10, 9, 8, 8, 8, 8, 7, 7]
'''

it_cell_gs = 67
it_ilu_inplace = 6
it_ilu_nocorr = [36, 33, 19, 14, 11, 10, 9, 8, 8, 8, 8, 7, 7]
it_ilu_corr = [41, 34, 20, 14, 11, 10, 9, 8, 8, 8, 8, 7, 7]

degrees = range(len(it_ilu_nocorr))

plt.plot(degrees, it_ilu_nocorr, 'x-', label='surrogate V1')
plt.plot(degrees, it_ilu_corr, 'x-', label='surrogate V2')
#plt.hlines(it_cell_gs, degrees[0], degrees[-1], linestyle='dotted', label='SGS')
plt.hlines(it_ilu_inplace, degrees[0], degrees[-1], linestyle='dashed', label='inplace')
plt.grid(True)
plt.xlabel('$dg_x = dg_y = dg_z$')
plt.ylabel('iterations')
plt.legend()
plt.show()

