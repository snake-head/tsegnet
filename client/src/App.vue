<template>
  <div id='root'>
    <el-container id="mainContainer">
        <el-aside id="listContainer">
          <List/>
        </el-aside>
        <el-container>
          <el-main>
            <Display/>
          </el-main>
          <el-footer id="operateContainer">
            <Operate/>
          </el-footer>
        </el-container>
    </el-container>
  </div>
</template>

<script>
  import List from './components/List'
  import Display from './components/Display'
  import Operate from './components/Operate'

export default {
  name: 'App',
  components: {
    List, Display, Operate
  }
}
</script>

<style>
  #mainContainer{
    height: 1000px; 
    border: 10px solid #eee
  }
  #listContainer{
     width: 400px;
     background-color: rgb(255, 255, 255); 
     border-right: 10px solid #eee;
  }
  #operateContainer{
    height: 200px;
    border-top: 10px solid #eee
  }
</style>
