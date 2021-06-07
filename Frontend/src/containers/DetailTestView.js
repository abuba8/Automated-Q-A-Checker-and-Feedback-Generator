import React from "react";
import axios from "axios";
import  {Card, Button} from 'antd'
import ExamForm from "../component/ExamForm";

class DetailTestView extends React.Component {
    state = {
        module: {}
    }
    componentDidMount()
    {
        const questionID = this.props.match.params.questionID;
        axios.get(`http://127.0.0.1:8000/api/test/${questionID}`)
        .then(res => {
            this.setState({
                module: res.data
            })
            console.log(res.data)
        })
    }
    handleDelete = (event) => {
        const questionID = this.props.match.params.questionID;
        axios.delete(`http://127.0.0.1:8000/api/module/${questionID}/delete`);
        this.props.history.push('/');
        this.forceUpdate();
    }
    render()
    { 
        var options = { year: 'numeric', month: 'long', day: 'numeric' };
        return( 
            <div>
               <Card title={this.state.module.title}>
                    <p>Name: {this.state.module.name}</p>
                    <p>Number of questions: {this.state.module.test_count}</p>
                    <p>Description of test: {this.state.module.description} </p>
                    <p>Type of question: {this.state.module.questiontype} </p>
                    <p>Date created: {new Date(this.state.module.created_date).toLocaleDateString([], options)} </p>
                </Card>
                <ExamForm requestMethod="put"
                            moduleID={this.props.match.params.moduleID}
                            questionID={this.props.match.params.questionID}
                            btnText="Update a Test Information!"/> 
                <form onSubmit={this.handleDelete}>
                    <Button type="danger" htmlType="submit">Delete</Button>
                </form>
            </div>
        )
    }
}
export default DetailTestView;