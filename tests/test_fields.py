import numpy as np



if __name__ == '__main__':
    x_values = np.linspace(-10, 10, 10).reshape(-1, 1)
    y_values = np.sin(x_values)
    # test_field = ScalarField(CSModel, (-10, 10))
    # test_field.fit_model(x_values, y_values)

    # new_x = np.linspace(-10, 10, 100).reshape(-1, 1)
    # new_y = test_field.predict_values(new_x)
    # grid = generate_grid(pos_2D, 30, 30)
    # new_temps = temp_field.predict_values(grid)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_trisurf(grid[:,0], grid[:,1], new_temps, cmap=cm.plasma, linewidth=0.1)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    # plt.close()

    # new_disps = disp_field.predict_values(grid)
    # mags = []
    # for v in new_disps:
    #     mags.append(np.linalg.norm(v))
    
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_trisurf(grid[:,0], grid[:,1], mags, cmap=cm.plasma, linewidth=0.1)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    # plt.close()
