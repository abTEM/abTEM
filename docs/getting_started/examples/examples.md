```{eval-rst}
:parenttoc: True

.. title:: Gallery: contents

===================
Example Gallery
===================

.. toctree::
   :hidden:

   cbed_quickstart
   hrtem_quickstart
   prism_quickstart
   hbn_dft_iam

```

Welcome to the `abTEM` gallery. These notebooks demonstrate specific a many of the concepts learned in the

* {bdg-success}`basic` Examples that cover the basics image simulation, these examples are generally appropriate for new
  users. Many of these examples may also be used as a template for performing a specific common type of simulation.
* {bdg-danger}`specialized` Examples that introduce a less common  Most of these examples assume that the user have
  experience with image simulation and abTEM, specifically.
* {bdg-primary}`publication` Examples that replicate published results, either fully or partially.

If you’d like to add your book to this list, simply add an entry to this gallery.yml file and open a Pull Request to add
it. For more detailed instructions see our guide on contributing to abTEM.

::::{grid} 3
:gutter: 0

:::{grid-item-card}  
:margin: 0
:link: examples:cbed_quickstart
:link-type: ref
:class-body: text-center
:class-header: bg-light text-center

**CBED**
^^^

```{glue:} cbed_quickstart
```

+++
{bdg-success}`quickstart`
:::

:::{grid-item-card}  
:margin: 0
:link: examples:hrtem_quickstart
:link-type: ref
:class-body: text-center
:class-header: bg-light text-center

**HRTEM**
^^^

```{glue:} hrtem_quickstart
```

+++
{bdg-success}`quickstart`
:::

:::{grid-item-card}  
:margin: 0
:link: examples:prism_quickstart
:link-type: ref
:class-body: text-center
:class-header: bg-light text-center



**PRISM**
^^^

```{glue:} prism_quickstart
```

+++
{bdg-success}`quickstart`
:::

:::{grid-item-card}  
:margin: 0
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
:margin: 0
:link: examples:hbn_dft_iam
:link-type: ref
:class-body: text-center
:class-header: bg-light text-center


**test**
^^^

```{glue:} prism_quickstart
```

+++
{bdg-primary}`publication`
:::

::::