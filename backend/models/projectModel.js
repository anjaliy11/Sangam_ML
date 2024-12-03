const mongoose = require('mongoose');

const ProjectSchema = new mongoose.Schema({
    project_id: { type: String, required: true },
    department: { type: String, required: true },
    task_priority: { type: Number, required: true },
    task_complexity: { type: Number, required: true },
    resources_allocated: { type: Number, required: true },
    communication_frequency: { type: Number, required: true },
    historical_delay: { type: Number, required: true },
    expected_completion_time: { type: Number, required: true },
    actual_completion_time: { type: Number },
    site_location: { type: String, required: true },
    conflict_indicator: { type: Boolean, required: true },
});

module.exports = mongoose.model('Project', ProjectSchema);
