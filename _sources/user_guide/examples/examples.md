(user_guide:example_gallery)=
# Example gallery

  ```{eval-rst}
.. toctree::
   :hidden:

   notebooks/cbed_quickstart
   notebooks/hrtem_quickstart
   notebooks/prism_quickstart
   notebooks/stem_quickstart
   notebooks/hbn_dft_iam
```

Welcome to the `abTEM` gallery. These notebooks demonstrate specific a many of the concepts learned in the

* {bdg-success}`basic` Examples that cover the basics image simulation, these examples are generally appropriate for new
  users. Many of these examples may also be used as a template for performing a specific common type of simulation.
* {bdg-danger}`specialized` Examples that introduce a less common topic in simulation of electron microscopy. Some of
  these examples assume that the user have experience abTEM.
* {bdg-primary}`publication` Examples that replicate published results, either fully or partially.

If you’d like to add your book to this list, simply add an entry to this gallery.yml file and open a Pull Request to add
it. For more detailed instructions see our guide on contributing to abTEM.

::::{grid} 3
:gutter: 2

:::{grid-item-card}
:link: examples:cbed_quickstart
:link-type: ref
:class-body: text-center
:class-header: bg-light text-center
**CBED**
^^^

```{glue:} cbed_quickstart
```

+++
{bdg-success}`basic`
:::




:::{grid-item-card}
:link: examples:hrtem_quickstart
:link-type: ref
:class-body: text-center
:class-header: bg-light text-center

**HRTEM**
^^^

```{glue:} hrtem_quickstart
```

+++
{bdg-success}`basic`
:::




:::{grid-item-card}
:link: examples:stem_quickstart
:link-type: ref
:class-body: text-center
:class-header: bg-light text-center

**STEM**
^^^

```{glue:} prism_quickstart
```

+++
{bdg-success}`basic`
:::



:::{grid-item-card}
:link: examples:prism_quickstart
:link-type: ref
:class-body: text-center
:class-header: bg-light text-center

**PRISM**
^^^

```{glue:} prism_quickstart
```

+++
{bdg-success}`basic`
:::

:::{grid-item-card}
:link: examples:hbn_dft_iam
:link-type: ref
:class-body: text-center
:class-header: bg-light text-center

**hBN with DFT potential**
^^^

```{glue:} hbn_dft_iam
```

+++
{bdg-primary}`publication`
:::

:::{grid-item-card}
:link: examples:hbn_dft_iam
:link-type: ref
:class-body: text-center
:class-header: bg-light text-center

**test**
^^^

```{glue:} prism_quickstart
```

+++
{bdg-danger}`specialized`
:::

::::