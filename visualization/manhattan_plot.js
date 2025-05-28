// Configuration
const config = {
    margin: { top: 40, right: 20, bottom: 60, left: 60 },
    height: 600,
    threshold: 5,
    colors: {
        significant: '#e41a1c',
        chromosome: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    }
};

// Initialize the visualization
function initManhattanPlot(data) {
    const width = document.getElementById('plot-container').clientWidth - config.margin.left - config.margin.right;
    const height = config.height - config.margin.top - config.margin.bottom;

    // Create SVG
    const svg = d3.select('#manhattan-plot')
        .append('svg')
        .attr('width', width + config.margin.left + config.margin.right)
        .attr('height', height + config.margin.top + config.margin.bottom)
        .append('g')
        .attr('transform', `translate(${config.margin.left},${config.margin.top})`);

    // Create scales
    const xScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.position)])
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.logP)])
        .range([height, 0]);

    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.5, 20])
        .on('zoom', (event) => {
            const newXScale = event.transform.rescaleX(xScale);
            const newYScale = event.transform.rescaleY(yScale);
            
            // Update points
            svg.selectAll('.dot')
                .attr('cx', d => newXScale(d.position))
                .attr('cy', d => newYScale(d.logP));
            
            // Update axes
            svg.select('.x-axis').call(d3.axisBottom(newXScale));
            svg.select('.y-axis').call(d3.axisLeft(newYScale));
            
            // Update threshold line
            svg.select('.threshold-line')
                .attr('y1', newYScale(config.threshold))
                .attr('y2', newYScale(config.threshold));
        });

    svg.call(zoom);

    // Add reset zoom button
    d3.select('.controls')
        .append('button')
        .attr('id', 'reset-zoom')
        .text('Reset Zoom')
        .on('click', () => {
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        });

    // Add axes
    svg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale));

    svg.append('g')
        .attr('class', 'y-axis')
        .call(d3.axisLeft(yScale));

    // Add axis labels
    svg.append('text')
        .attr('class', 'x-label')
        .attr('x', width / 2)
        .attr('y', height + 40)
        .style('text-anchor', 'middle')
        .text('Genomic Position');

    svg.append('text')
        .attr('class', 'y-label')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -40)
        .style('text-anchor', 'middle')
        .text('-log10(p-value)');

    // Add threshold line
    const thresholdLine = svg.append('line')
        .attr('class', 'threshold-line')
        .attr('x1', 0)
        .attr('x2', width)
        .attr('y1', yScale(config.threshold))
        .attr('y2', yScale(config.threshold))
        .style('stroke', '#ff0000')
        .style('stroke-dasharray', '5,5');

    // Add chromosome filter
    const chromosomes = [...new Set(data.map(d => d.chromosome))].sort((a, b) => a - b);
    const chromosomeFilter = d3.select('.controls')
        .append('div')
        .attr('class', 'chromosome-filter');

    chromosomeFilter.append('label')
        .text('Chromosomes: ');

    const chromosomeCheckboxes = chromosomeFilter
        .append('div')
        .attr('class', 'checkbox-group');

    chromosomes.forEach(chr => {
        chromosomeCheckboxes.append('label')
            .append('input')
            .attr('type', 'checkbox')
            .attr('value', chr)
            .attr('checked', true)
            .on('change', updatePlot);
    });

    // Add search functionality
    const searchDiv = d3.select('.controls')
        .append('div')
        .attr('class', 'search-control');

    searchDiv.append('label')
        .text('Search Variant/Gene: ');

    searchDiv.append('input')
        .attr('type', 'text')
        .attr('id', 'search-input')
        .attr('placeholder', 'Enter variant ID or gene name')
        .on('input', debounce(handleSearch, 300));

    // Add points
    const points = svg.selectAll('.dot')
        .data(data)
        .enter()
        .append('circle')
        .attr('class', 'dot')
        .attr('cx', d => xScale(d.position))
        .attr('cy', d => yScale(d.logP))
        .attr('r', 3)
        .style('fill', d => d.logP >= config.threshold ? config.colors.significant : config.colors.chromosome[d.chromosome % 10])
        .on('mouseover', function(event, d) {
            d3.select(this)
                .attr('r', 5);
            
            const tooltip = d3.select('#tooltip');
            tooltip.html(`
                <strong>Variant:</strong> ${d.variant}<br>
                <strong>Gene:</strong> ${d.gene}<br>
                <strong>P-value:</strong> ${d.pValue.toExponential(2)}<br>
                <strong>Effect Size:</strong> ${d.effectSize.toFixed(3)}<br>
                <strong>Chromosome:</strong> ${d.chromosome}<br>
                <strong>Position:</strong> ${d.position}
            `)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .classed('visible', true);
        })
        .on('mouseout', function() {
            d3.select(this)
                .attr('r', 3);
            d3.select('#tooltip')
                .classed('visible', false);
        });

    // Add threshold control
    d3.select('#threshold')
        .on('input', function() {
            const newThreshold = +this.value;
            config.threshold = newThreshold;
            thresholdLine
                .attr('y1', yScale(newThreshold))
                .attr('y2', yScale(newThreshold));
            
            points.style('fill', d => d.logP >= newThreshold ? config.colors.significant : config.colors.chromosome[d.chromosome % 10]);
        });

    // Update plot based on chromosome filter
    function updatePlot() {
        const selectedChromosomes = Array.from(document.querySelectorAll('.chromosome-filter input:checked'))
            .map(input => +input.value);

        points.style('opacity', d => selectedChromosomes.includes(d.chromosome) ? 1 : 0.1);
    }

    // Handle search
    function handleSearch() {
        const searchTerm = d3.select('#search-input').property('value').toLowerCase();
        
        points.style('opacity', d => {
            if (!searchTerm) return 1;
            return (d.variant.toLowerCase().includes(searchTerm) || 
                    d.gene.toLowerCase().includes(searchTerm)) ? 1 : 0.1;
        });
    }

    // Debounce function for search
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Example data structure
const exampleData = [
    {
        variant: "rs123456",
        gene: "GENE1",
        chromosome: 1,
        position: 1000000,
        pValue: 1e-6,
        effectSize: 0.5,
        logP: 6
    },
    // Add more data points here
];

// Load and process data
function loadData(file) {
    d3.csv(file).then(data => {
        // Process the data
        const processedData = data.map(d => ({
            variant: d.variant,
            gene: d.gene,
            chromosome: +d.chromosome,
            position: +d.position,
            pValue: +d.pValue,
            effectSize: +d.effectSize,
            logP: -Math.log10(+d.pValue)
        }));
        
        initManhattanPlot(processedData);
    });
}

// Initialize with example data
initManhattanPlot(exampleData); 