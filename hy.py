For  Gi in Generators:
Input: XGi: (B, Wi + 1, D)  # Input to the generator, shape (Batch size B, Window size Wi+1, Feature dimension D)

# Generator outputs
Regression Output: Ryi: (B, 1)  # Regression task output, shape (Batch size B, 1)
Classification Output: Cyi: (B, 1)  # Classification task output, shape (Batch size B, 1)

Output: Yi = Ryi, Cyi # Generator's output is the combination of regression and classification results

For (Ryi, Cyi) in Yi:
    for windowj in windows
        XDi<-concat(realY)


For each epoch:

 # Iterate over each generator's output
XDi = (y_
       For each discriminator Dj in Discriminators:  # For each discriminator
       If epoch % 2 == 1:
       LossDjReal=-log(Dj())





1.
Input: Yi = (Ryi, Cyi)  # Generator's output (Regression + Classification)
# Discriminator Training
2.
Discriminator
Dj
receives
Yi(generated
output) and makes
a
prediction:
- Compute
the
regression
loss:
`L_D_regression = BCELoss(Dj(Ryi), true
regression
labels)`
- Compute
the
classification
loss:
`L_D_classification = BCELoss(Dj(Cyi), true
classification
labels)`
- Total
discriminator
loss: `L_D = L_D_regression + L_D_classification
`
- Backpropagate and update
discriminator
's parameters:
3.
`optimizer_Dj.zero_grad()`  # Clear the gradients
4.
`L_D.backward()`  # Backpropagate gradients
5.
`optimizer_Dj.step()`  # Update discriminator's parameters

# Generator Training
3.
The
generator
wants
to
minimize
the
discriminator’s
incorrect
classification
of
its
output:
- Compute
the
regression
loss
for the generator:
    `L_G_regression = BCEWithLogitsLoss(Dj(Ryi), true_labels)
`
- Compute
the
classification
loss
for the generator:
    `L_G_classification = BCEWithLogitsLoss(Dj(Cyi), true_labels)
`
- Total
generator
loss: `L_G = L_G_regression + L_G_classification
`
- Backpropagate and update
generator
's parameters:
4.
`optimizer_Gi.zero_grad()`  # Clear the gradients
5.
`L_G.backward()`  # Backpropagate gradients
6.
`optimizer_Gi.step()`  # Update generator's parameters

# Update both generator and discriminator's parameters using backpropagation
